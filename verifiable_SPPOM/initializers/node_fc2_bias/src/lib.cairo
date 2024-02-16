use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_fc2_bias() -> Tensor<FP16x16> {
    let mut shape = array![25];

    let mut data = array![FP16x16 { mag: 169693, sign: true }, FP16x16 { mag: 104239, sign: false }, FP16x16 { mag: 32446, sign: false }, FP16x16 { mag: 14695, sign: false }, FP16x16 { mag: 12548, sign: false }, FP16x16 { mag: 41431, sign: false }, FP16x16 { mag: 23393, sign: false }, FP16x16 { mag: 20711, sign: true }, FP16x16 { mag: 7580, sign: false }, FP16x16 { mag: 8101, sign: false }, FP16x16 { mag: 36125, sign: true }, FP16x16 { mag: 1820, sign: false }, FP16x16 { mag: 15308, sign: true }, FP16x16 { mag: 57691, sign: false }, FP16x16 { mag: 3989, sign: true }, FP16x16 { mag: 54887, sign: false }, FP16x16 { mag: 6394, sign: false }, FP16x16 { mag: 174348, sign: true }, FP16x16 { mag: 10578, sign: true }, FP16x16 { mag: 1187, sign: false }, FP16x16 { mag: 10639, sign: false }, FP16x16 { mag: 55202, sign: false }, FP16x16 { mag: 20057, sign: true }, FP16x16 { mag: 11722, sign: false }, FP16x16 { mag: 20994, sign: false }];

    TensorTrait::new(shape.span(), data.span())
}