# Phi-3 Inference in Rust

This project provides Phi-3 inference using [candle](https://github.com/huggingface/candle),
provided both as a library and a binary, see the sections below for examples of how to use both.

The project is setup as a Cargo workspace so the binary can have shared and independent dependencies
with the library.

This project is simply a learning experiment and I don't plan on maintaining it beyond my personal use.

## Using the GPU

CUDA support is not compiled by default, to enable it use the feature flag `cuda`.

```bash
# To compile the binary with CUDA support
cargo build --release --features cuda
# And to run it directly with Cargo
cargo run --features cuda
```

## Library Usage

A simple example using the default configuration options:

```rust
use phi3::{Phi3, Phi3Config};

fn main() {
    let mut phi = Phi3::init(Phi3Config::default())
        .expect("failed to initialize and prepare model weights and tokenizer");
    let prompt = "<|user|>\nHelp me I'm dying!<|end|>\n<|assistant|>\n";
    let output = phi.complete(prompt).expect("failed to run inference");
}
```

With custom configuration and using GPU via CUDA:

```rust
use phi3::{Phi3, Phi3Config};
use candle_core::Device;

fn main() {
    let gpu_device = Device::new_cuda(0)
        .expect("couldn't access CUDA device, ensure you have CUDA installed and a supported graphics card connected");

    let config = Phi3Config {
        temperature: Some(0.8),
        sample_len: 1000,
        seed: 98374289,
        device: gpu_device,
        ..Default::default()
    };

    let mut phi = Phi3::init(config);
    let prompt = "<|user|>\nTell me a bedtime story<|end>\n<|assistant|>\nOnce upon a time, ";
    let output = phi.complete(prompt).expect("failed to run inference on GPU");
    println!("Once upon a time, {output}");
}
```

## CLI Usage

The CLI is well documented so you can simply use `phi3 --help` for usage, it includes the main
`complete` command for running inference and a `download` command to conveniently download the
model weights and tokenizer from Hugging Face.

```
$ phi3 --help
A simple CLI to run Phi-3 inference locally

Usage: phi3 <COMMAND>

Commands:
  complete  Run inference on a model completing a given prompt
  download  Download the model and tokenizer from Hugging Face if not already available
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

The complete command has quite a few arguments but they are all optional expect prompt which is the
only positional argument:

```
$ phi3 complete --help
Run inference on a model completing a given prompt

Usage: phi3 complete [OPTIONS] <PROMPT>

Arguments:
  <PROMPT>  The prompt for Phi-3 to complete. Prompts are recommended to follow a specific format, see https://huggingface.co/microsoft/Phi-3-mini-128k-instruct#chat-format for more details:

Options:
      --sample-length <SAMPLE_LENGTH>
          The max tokens to output. If the output tokens reaches the sample_length inference will stop immediately and return the output, potentially even stopping mid word
      --temperature <TEMPERATURE>
          The randomness of the output, must be a number between 0.0 and 1.0, where 0.0 is deterministic and 1.0 output will vary drastically. This should NOT be used in conjunction with `top_p`, they are exclusive of each other
      --top-p <TOP_P>
          Used for nucleus sampling to limit the number of tokens considered for each step. Most of the time temperature is all you'll need, this is for advanced use-cases. This should NOT be used in conjunction with `temperature`, they are exclusive of each other
      --repeat-last-n <REPEAT_LAST_N>
          Controls the number of tokens considered for applying a penalty to prevent repetitive sequences
      --repeat-penalty <REPEAT_PENALTY>
          The penalty value to discourage repeated tokens
      --seed <SEED>
          A random seed applied when temperature is not 0.0, the same value should reproduce the same output
      --cpu
          Runs inference on the CPU, this is the default. Only `--cpu` or `--gpu` should be provided, GPU will take priority if both are true
      --gpu
          Runs inference on the GPU, the default is CPU. Only `--gpu` or `--cpu` should be provided, GPU will take priority if both are enabled
  -h, --help
          Print help
```
