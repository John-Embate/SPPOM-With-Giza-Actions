use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_fc3_weight() -> Tensor<FP16x16> {
    let mut shape = array![1, 25];

    let mut data = array![FP16x16 { mag: 126075, sign: true }, FP16x16 { mag: 58740, sign: false }, FP16x16 { mag: 40940, sign: false }, FP16x16 { mag: 35526, sign: false }, FP16x16 { mag: 18260, sign: false }, FP16x16 { mag: 33980, sign: false }, FP16x16 { mag: 22777, sign: false }, FP16x16 { mag: 31018, sign: false }, FP16x16 { mag: 10855, sign: true }, FP16x16 { mag: 4241, sign: false }, FP16x16 { mag: 20272, sign: true }, FP16x16 { mag: 7865, sign: false }, FP16x16 { mag: 39592, sign: false }, FP16x16 { mag: 94467, sign: false }, FP16x16 { mag: 1128, sign: false }, FP16x16 { mag: 55589, sign: false }, FP16x16 { mag: 12002, sign: true }, FP16x16 { mag: 102823, sign: true }, FP16x16 { mag: 19708, sign: true }, FP16x16 { mag: 21309, sign: false }, FP16x16 { mag: 18152, sign: false }, FP16x16 { mag: 66797, sign: true }, FP16x16 { mag: 11486, sign: true }, FP16x16 { mag: 16913, sign: false }, FP16x16 { mag: 26118, sign: false }];

    TensorTrait::new(shape.span(), data.span())
}