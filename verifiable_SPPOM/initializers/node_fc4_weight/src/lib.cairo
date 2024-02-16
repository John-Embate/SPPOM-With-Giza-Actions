use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_fc4_weight() -> Tensor<FP16x16> {
    let mut shape = array![1, 25];

    let mut data = array![FP16x16 { mag: 19166, sign: false }, FP16x16 { mag: 12080, sign: true }, FP16x16 { mag: 93400, sign: true }, FP16x16 { mag: 19486, sign: false }, FP16x16 { mag: 4165, sign: true }, FP16x16 { mag: 79003, sign: true }, FP16x16 { mag: 10997, sign: false }, FP16x16 { mag: 58236, sign: false }, FP16x16 { mag: 65065, sign: true }, FP16x16 { mag: 52183, sign: false }, FP16x16 { mag: 20594, sign: false }, FP16x16 { mag: 10847, sign: true }, FP16x16 { mag: 24845, sign: false }, FP16x16 { mag: 68877, sign: false }, FP16x16 { mag: 2217, sign: false }, FP16x16 { mag: 32273, sign: false }, FP16x16 { mag: 111791, sign: true }, FP16x16 { mag: 261688, sign: false }, FP16x16 { mag: 12219, sign: true }, FP16x16 { mag: 8715, sign: false }, FP16x16 { mag: 93154, sign: true }, FP16x16 { mag: 32388, sign: false }, FP16x16 { mag: 42156, sign: true }, FP16x16 { mag: 14731, sign: false }, FP16x16 { mag: 119430, sign: true }];

    TensorTrait::new(shape.span(), data.span())
}