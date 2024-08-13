use phi3::{Phi3, Phi3Config};

fn main() {
    let mut phi3 = Phi3::init(Phi3Config::default()).expect("failed to initialize Phi3");
    let output = phi3
        .complete("<|user|>\nWrite a haiku about rust software dev<|end|>\n<|assistant|>\n")
        .expect("failed to run Phi3 inference");
    println!("output: {output}");
}
