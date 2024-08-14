use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::{
    generation::LogitsProcessor, models::quantized_phi3::ModelWeights as Phi3Model,
};
use tokenizers::Tokenizer;

/// Contains all possible errors that can occur in this library for better error handling and
/// debugging.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to get last token, make sure the prompt is not empty")]
    MissingLastToken,

    #[error("tokenizer is missing eos token, can't safely continue")]
    MissingEosToken,

    #[error("std IO error: {0:?}")]
    Io(#[from] std::io::Error),

    #[error("hugging face API error: {0:?}")]
    HfHubApi(#[from] hf_hub::api::sync::ApiError),

    #[error("candle-core error: {0:?}")]
    CandleCore(#[from] candle_core::Error),

    #[error("tokenizer error: {0:?}")]
    Tokenizer(#[from] tokenizers::Error),

    #[error("unknown boxed error: {0:?}")]
    Box(#[from] Box<dyn std::error::Error>),
}

/// A convenience type wrapping the standard Result with our custom Error.
pub type Result<T> = std::result::Result<T, Error>;

/// Contains all available configuration options for running Phi-3 inference.
///
/// # Examples
///
/// ```
/// use phi3::Phi3Config;
/// use candle_core::Device;
///
/// let config = Phi3Config {
///     temperature: Some(0.5),
///     sample_len: 500,
///     device: Device::Cpu,
///     ..Default::default()
/// }
/// ```
///
/// # Future Additions
///
/// A builder may be implemented for another option when constructing the config.
#[derive(Debug, Clone)]
pub struct Phi3Config {
    /// Specifies the maximum length of the generated text.
    pub sample_len: usize,

    /// Controls the randomness of the sampling process, either `temperature` or `top_p` is
    /// required but temperature is most likely what you'll want except for advanced cases.
    pub temperature: Option<f64>,

    /// Used for nucleus sampling to limit the number of tokens considered for each step, most of
    /// the time temperature is all you'll need, this is for advanced use-cases.
    pub top_p: Option<f64>,

    /// Controls the number of tokens considered for applying a penalty to prevent repetitive
    /// sequences.
    pub repeat_last_n: usize,

    /// The penalty value to discourage repeated tokens.
    pub repeat_penalty: f32,

    /// A random seed applied when temperature is not 0.0, the same value should reproduce the same
    /// output.
    pub seed: u64,

    /// Can be `Device::Cpu` (no extra dependencies) or `Device::Gpu` (which requires CUDA).
    pub device: Device,
}

impl Default for Phi3Config {
    fn default() -> Self {
        Self {
            temperature: Some(0.0),
            sample_len: 100,
            top_p: None,
            repeat_last_n: 64,
            repeat_penalty: 1.2,
            // Some arbitrary number, this could be any u64 value
            seed: 1000000,
            device: Device::Cpu,
        }
    }
}

/// Provides everything necessary to run inference given a configuration struct.
pub struct Phi3 {
    config: Phi3Config,
    model: Phi3Model,
    tokenizer: Tokenizer,
}

impl Phi3 {
    /// This function will download the Phi-3 model weights and tokenizer if they aren't already
    /// available, this can take quite a while even though it's a small model.
    ///
    /// When using this in a production environment weights and the tokenizer should ideally be
    /// downloaded and packaged along size the compiled binary to prevent slow start up times.
    pub fn init(config: Phi3Config) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        // Download the model weights if it isn't already available locally.
        let model_path = api
            .repo(hf_hub::Repo::with_revision(
                "microsoft/Phi-3-mini-4k-instruct-gguf".to_string(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ))
            .get("Phi-3-mini-4k-instruct-q4.gguf")?;

        // Download the tokenizer if it isn't already available locally.
        let tokenizer_path = api
            .model("microsoft/Phi-3-mini-4k-instruct".to_string())
            .get("tokenizer.json")?;

        // Load the models from the downloaded file.
        let mut file = std::fs::File::open(&model_path)?;
        let model_content = gguf_file::Content::read(&mut file)?;
        let model = Phi3Model::from_gguf(false, model_content, &mut file, &config.device)?;

        // Load the tokenizer from the downloaded file.
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        Ok(Self {
            config,
            tokenizer,
            model,
        })
    }

    /// Run inference given a prompt, stops when `sample_len` in `Phi3Config` is reached or EOS
    /// token is output.
    ///
    /// # Example
    ///
    /// ```
    /// use phi3::{Phi3, Phi3Config};
    ///
    /// // Because of the training data used in Phi-3 the following prompt structure is
    /// // recommended for the best results.
    /// let prompt = "<|user|>\nWrite a haiku about Rust software dev<|end|>\n<|assistant|>\n";
    /// let mut phi = Phi3::init(Phi3Config::default()).expect("failed to initialize");
    /// let output = phi.complete(prompt).expect("failed to run inference");
    /// println!("{output}");
    /// ```
    ///
    /// # Future Improvements
    ///
    /// This should return a stream so users don't have to wait for the entire sequence to finish
    /// to start using the output.
    pub fn complete(&mut self, prompt: &str) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt, true)?;
        let tokens = tokens.get_ids();
        let to_sample = self.config.sample_len.saturating_sub(1);
        let mut all_tokens = vec![];

        let mut logits_processor =
            LogitsProcessor::new(self.config.seed, self.config.temperature, self.config.top_p);

        let mut next_token = *tokens.last().ok_or(Error::MissingLastToken)?;
        let eos_token = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or(Error::MissingEosToken)?;
        let mut prev_text_len = 0;

        for (pos, &token) in tokens.iter().enumerate() {
            let input = Tensor::new(&[token], &self.config.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;

            if pos == tokens.len() - 1 {
                next_token = logits_processor.sample(&logits)?;
                all_tokens.push(next_token);
            }
        }

        let mut output_text = String::new();

        // Run inference until `sample_len` is reached or EOS token is output.
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &self.config.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if self.config.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(self.config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.config.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };

            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);

            let decoded_text = self.tokenizer.decode(&all_tokens, true)?;

            if decoded_text.len() > prev_text_len {
                let new_text = &decoded_text[prev_text_len..];
                output_text.push_str(new_text);
                prev_text_len = decoded_text.len();
            }

            if next_token == eos_token {
                break;
            }
        }

        Ok(output_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keep a consistent configuration across tests for deterministic results.
    fn config() -> Phi3Config {
        Phi3Config {
            temperature: Some(0.0),
            sample_len: 100,
            top_p: None,
            repeat_last_n: 64,
            repeat_penalty: 1.2,
            seed: 1000000,
            device: Device::Cpu,
        }
    }

    #[test]
    fn outputs_as_expected() {
        let prompt = r#"
        <|user|>
        1 + 1<|end|>
        <|assistant|>
        2<|end|>
        <|user|>
        1 + 5<|end|>
        <|assistant|>
        6<|end|>
        <|user|>
        2 + 10<|end|>
        <|assistant|>
        "#.trim_start();

        let mut config = config();
        // Struggled to find a prompt that would be short where the model doesn't keep going
        // excessively so setting the sample length will only get the answer we expect.
        config.sample_len = 3;
        let mut phi3 = Phi3::init(config).expect("failed to init");
        let output = phi3.complete(prompt).expect("failed to run inference");
        assert_eq!(&output, "12");
    }

    #[test]
    fn stops_when_sample_len_is_reached() {
        // Set a tiny sample length for faster testing.
        let mut config = config();
        config.sample_len = 4;

        let prompt = "<|user|>\nTell me a story<|end|>\n<|assistant|>\nOnce upon a time, ";

        let mut phi3 = Phi3::init(config).expect("failed to init");
        let output = phi3.complete(prompt).expect("failed to run inference");
        assert_eq!(&output, "in a small village");
    }
}
