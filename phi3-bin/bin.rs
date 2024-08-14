use candle_core::Device;
use clap::{Args, Parser, Subcommand};
use phi3::{Phi3, Phi3Config};

/// A simple CLI to run Phi-3 inference locally.
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// The main command to execute.
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    /// Run inference on a model completing a given prompt.
    Complete(CompleteArgs),

    /// Download the model and tokenizer from Hugging Face if not already available.
    ///
    /// The same download process will happen when using the `run` command, this is a convenient
    /// way to only download them when you don't want to run inference in the same command.
    Download,
}

#[derive(Args)]
struct CompleteArgs {
    /// The prompt for Phi-3 to complete.
    /// Prompts are recommended to follow a specific format, see
    /// https://huggingface.co/microsoft/Phi-3-mini-128k-instruct#chat-format for more details:
    prompt: String,

    /// The max tokens to output.
    /// If the output tokens reaches the sample_length inference will stop immediately and return
    /// the output, potentially even stopping mid word.
    #[arg(long)]
    sample_length: Option<usize>,

    /// The randomness of the output, must be a number between 0.0 and 1.0, where 0.0 is
    /// deterministic and 1.0 output will vary drastically.
    /// This should NOT be used in conjunction with `top_p`, they are exclusive of each other.
    #[arg(long)]
    temperature: Option<f64>,

    /// Used for nucleus sampling to limit the number of tokens considered for each step.
    /// Most of the time temperature is all you'll need, this is for advanced use-cases.
    /// This should NOT be used in conjunction with `temperature`, they are exclusive of each other.
    #[arg(long)]
    top_p: Option<f64>,

    /// Controls the number of tokens considered for applying a penalty to prevent repetitive
    /// sequences.
    #[arg(long)]
    repeat_last_n: Option<usize>,

    /// The penalty value to discourage repeated tokens.
    #[arg(long)]
    repeat_penalty: Option<f32>,

    /// A random seed applied when temperature is not 0.0, the same value should reproduce the same
    /// output.
    #[arg(long)]
    seed: Option<u64>,

    /// Runs inference on the CPU, this is the default.
    /// Only `--cpu` or `--gpu` should be provided, GPU will take priority if both are true.
    #[arg(long)]
    cpu: bool,

    /// Runs inference on the GPU, the default is CPU.
    /// Only `--gpu` or `--cpu` should be provided, GPU will take priority if both are enabled.
    #[arg(long)]
    gpu: bool,
}

impl Into<Phi3Config> for CompleteArgs {
    fn into(self) -> Phi3Config {
        let mut config = Phi3Config::default();

        if let Some(sample_len) = self.sample_length {
            config.sample_len = sample_len;
        }

        // If `top_p` is set it should override `temperature`
        if let Some(top_p) = self.top_p {
            config.top_p = Some(top_p);
            config.temperature = None;
        } else if let Some(temp) = self.temperature {
            config.temperature = Some(temp);
        }

        if let Some(repeat_last_n) = self.repeat_last_n {
            config.repeat_last_n = repeat_last_n;
        }

        if let Some(repeat_penalty) = self.repeat_penalty {
            config.repeat_penalty = repeat_penalty;
        }

        if let Some(seed) = self.seed {
            config.seed = seed;
        }

        if self.gpu {
            config.device = Device::new_cuda(0)
                .expect("gpu enabled but couldn't access CUDA device, ensure you have CUDA installed and a supported graphics card connected");
        } else {
            config.device = Device::Cpu;
        }

        config
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        CliCommand::Complete(args) => {
            let prompt = args.prompt.clone();
            let mut phi3 = Phi3::init(args.into()).expect("failed to initialize Phi3");
            let output = phi3
                .complete(&prompt)
                .expect("failed to run Phi3 inference");
            println!("{output}");
        }
        CliCommand::Download => {
            Phi3::init(Phi3Config::default())
                .expect("failed to download and verify model weights and tokenizer");
            println!("done")
        }
    }
}
