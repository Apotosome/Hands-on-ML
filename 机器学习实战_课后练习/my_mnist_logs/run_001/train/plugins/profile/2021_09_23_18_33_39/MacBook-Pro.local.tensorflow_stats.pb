"?L
BHostIDLE"IDLE1??ʡ???@A??ʡ???@a-?ĒB??i-?ĒB???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1?MbXݞ@9?MbXݞ@A?MbXݞ@I?MbXݞ@a#PW???iq??=?????Unknown?
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1     P{@9     P{@A     P{@I     P{@a??:eծ??i????({???Unknown
sHost_FusedMatMul"sequential_1/dense_3/Relu(1     ?s@9     ?s@A     ?s@I     ?s@aJ?iEي?i??|??????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_4/MatMul(1     @^@9     @^@A     @^@I     @^@a'?S<?t?i?k?v????Unknown
HostMatMul"+gradient_tape/sequential_1/dense_4/MatMul_1(1     ?]@9     ?]@A     ?]@I     ?]@al??E?Yt?i,ѣ8???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      R@9      R@A      R@I      R@a????%?h?i!??CQ???Unknown
sHost_FusedMatMul"sequential_1/dense_4/Relu(1     ?M@9     ?M@A     ?M@I     ?M@a????-d?iԛ??qe???Unknown
i	HostWriteSummary"WriteSummary(1^?IL@9^?IL@A^?IL@I^?IL@a?`?-?9c?i5????x???Unknown?
^
HostGatherV2"GatherV2(1     ?J@9     ?J@A     ?J@I     ?J@a??o? b?iTC(̊???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      C@9      C@A      C@I      C@awW׌`?Y?i ?nM˗???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      3@9      3@A      3@I      3@awW׌`?I?i????J????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      2@I      2@a????%?H?i????r????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_5/MatMul(1      ,@9      ,@A      ,@I      ,@a?h%?9'C?i?ga?<????Unknown
qHostSoftmax"sequential_1/dense_5/Softmax(1      ,@9      ,@A      ,@I      ,@a?h%?9'C?iG1̋????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      *@9      *@A      *@I      *@a?5???A?i?~x?x????Unknown
HostMatMul"+gradient_tape/sequential_1/dense_5/MatMul_1(1      *@9      *@A      *@I      *@a?5???A?i??$?????Unknown
dHostDataset"Iterator::Model(1     ?G@9     ?G@A      "@I      "@a????%?8?i?(??????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      "@9      "@A      "@I      "@a????%?8?i???????Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a????%?8?if?;'????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?h%?9'3?iGq ?????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      *@9      *@A      @I      @a?h%?9'3?i?????????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a?h%?9'3?im??U????Unknown
eHost
LogicalAnd"
LogicalAnd(1??"???@9??"???@A??"???@I??"???@a%???hw2?ii????????Unknown?
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?D??j0?i??n?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?D??j0?i?z?l?????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?D??j0?i%c\??????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?D??j0?i?K??????Unknown
vHost_FusedMatMul"sequential_1/dense_5/BiasAdd(1      @9      @A      @I      @a?D??j0?iM4Jv?????Unknown
vHostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @a?D??j0?i????????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?D@9     ?D@A      @I      @ab?ǆ?\+?i]?y??????Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @ab?ǆ?\+?i??1b`????Unknown
Z!HostArgMax"ArgMax(1      @9      @A      @I      @a?.???%?i<?+??????Unknown
t"HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?.???%?i??%?????Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?.???%?i?{????Unknown
|$HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a?.???%?ie?N?????Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?.???%?iȧ?7????Unknown
?&HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?.???%?i+?ĕ????Unknown
V'HostSum"Sum_2(1      @9      @A      @I      @a?.???%?i????????Unknown
?(HostReluGrad"+gradient_tape/sequential_1/dense_3/ReluGrad(1      @9      @A      @I      @a?.???%?i?x:R????Unknown
?)HostBiasAddGrad"6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?.???%?iTi?t?????Unknown
?*HostReadVariableOp"+sequential_1/dense_4/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?.???%?i?Y??????Unknown
?+HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1      @9      @A      @I      @a?.???%?iJ??l????Unknown
?,HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?.???%?i}:?%?????Unknown
V-HostCast"Cast(1      @9      @A      @I      @a?D??j ?iǮ$??????Unknown
?.HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??a?D??j ?i#`~?????Unknown
`/HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?D??j ?i[??*?????Unknown
u0HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a?D??j ?i????????Unknown
?1HostReluGrad"+gradient_tape/sequential_1/dense_4/ReluGrad(1      @9      @A      @I      @a?D??j ?i???????Unknown
?2HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?D??j ?i9?M/?????Unknown
?3HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?D??j ?i?h???????Unknown
?4HostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @a?D??j ?i??ć ????Unknown
?5HostReadVariableOp"+sequential_1/dense_5/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?D??j ?iQ 4????Unknown
?6HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1      @9      @A      @I      @a?D??j ?ia?;?????Unknown
?7HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      @9      @A      @I      @a?D??j ?i?9w?????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?.????i?1???????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?.????i*q?r????Unknown
X:HostCast"Cast_2(1       @9       @A       @I       @a?.????i>"??!????Unknown
X;HostEqual"Equal(1       @9       @A       @I       @a?.????iok?????Unknown
?<HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      .@9      .@A       @I       @a?.????i???????Unknown
u=HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a?.????i?
e=/????Unknown
b>HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?.????i?Z?????Unknown
??HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?.????i3?^x?????Unknown
?@HostReadVariableOp"*sequential_1/dense_4/MatMul/ReadVariableOp(1       @9       @A       @I       @a?.????id?ە<????Unknown
?AHostReadVariableOp"*sequential_1/dense_5/MatMul/ReadVariableOp(1       @9       @A       @I       @a?.????i??X??????Unknown
?BHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a?.????i???К????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?.????i?_?_?????Unknown
XDHostCast"Cast_3(1      ??9      ??A      ??I      ??a?.????i??R?I????Unknown
XEHostCast"Cast_4(1      ??9      ??A      ??I      ??a?.????iX}?????Unknown
TFHostMul"Mul(1      ??9      ??A      ??I      ??a?.????i*???????Unknown
sGHostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a?.????iCP??P????Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?.????i\?L)?????Unknown
yIHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?.????iuH??????Unknown
?JHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?.????i???FW????Unknown
?KHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?.????i?@?ծ????Unknown
aLHostIdentity"Identity(1D?l?????9D?l?????AD?l?????ID?l?????a?????J?i?????????Unknown?
JMHostReadVariableOp"div_no_nan_1/ReadVariableOp(i?????????Unknown*?K
uHostFlushSummaryWriter"FlushSummaryWriter(1?MbXݞ@9?MbXݞ@A?MbXݞ@I?MbXݞ@a?!]K???i?!]K????Unknown?
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1     P{@9     P{@A     P{@I     P{@a??x?'???iPV?????Unknown
sHost_FusedMatMul"sequential_1/dense_3/Relu(1     ?s@9     ?s@A     ?s@I     ?s@ar?p????iH?a??????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_4/MatMul(1     @^@9     @^@A     @^@I     @^@a颦?t??iٺˎ?????Unknown
HostMatMul"+gradient_tape/sequential_1/dense_4/MatMul_1(1     ?]@9     ?]@A     ?]@I     ?]@a7Sk??*??ip?ء????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      R@9      R@A      R@I      R@a)$?'?Ŕ?i-????d???Unknown
sHost_FusedMatMul"sequential_1/dense_4/Relu(1     ?M@9     ?M@A     ?M@I     ?M@aL?ϕ???io?m??????Unknown
iHostWriteSummary"WriteSummary(1^?IL@9^?IL@A^?IL@I^?IL@aZ??7??i:l?f?n???Unknown?
^	HostGatherV2"GatherV2(1     ?J@9     ?J@A     ?J@I     ?J@a?
????id|j?????Unknown
s
HostDataset"Iterator::Model::ParallelMapV2(1      C@9      C@A      C@I      C@a?{?TS???iR~?&?@???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      3@9      3@A      3@I      3@a?{?TS?u?iIe͠l???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      2@I      2@a)$?'??t?i?ô?,????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_5/MatMul(1      ,@9      ,@A      ,@I      ,@a??(t"(p?i??|????Unknown
qHostSoftmax"sequential_1/dense_5/Softmax(1      ,@9      ,@A      ,@I      ,@a??(t"(p?i?f??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      *@9      *@A      *@I      *@a?ޔ?dn?i????????Unknown
HostMatMul"+gradient_tape/sequential_1/dense_5/MatMul_1(1      *@9      *@A      *@I      *@a?ޔ?dn?ik???????Unknown
dHostDataset"Iterator::Model(1     ?G@9     ?G@A      "@I      "@a)$?'??d?i?2?ɕ'???Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      "@9      "@A      "@I      "@a)$?'??d?i????[<???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a)$?'??d?i?v?!Q???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a??(t"(`?i????Ia???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      *@9      *@A      @I      @a??(t"(`?ie??qq???Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a??(t"(`?i,?u??????Unknown
eHost
LogicalAnd"
LogicalAnd(1??"???@9??"???@A??"???@I??"???@a-B4?'_?iM???-????Unknown?
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a70?4??[?iej?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a70?4??[?i}??N?????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a70?4??[?i?Bߐ?????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a70?4??[?i???Ғ????Unknown
vHost_FusedMatMul"sequential_1/dense_5/BiasAdd(1      @9      @A      @I      @a70?4??[?i?l????Unknown
vHostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @a70?4??[?i݆.WE????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?D@9     ?D@A      @I      @a??^??W?iF6???????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??^??W?i???Z????Unknown
Z HostArgMax"ArgMax(1      @9      @A      @I      @azu??wR?ij??????Unknown
t!HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @azu??wR?i%?}????Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @azu??wR?i???????Unknown
|#HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @azu??wR?i??K H ???Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @azu??wR?iV????)???Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @azu??wR?i?#?2???Unknown
V&HostSum"Sum_2(1      @9      @A      @I      @azu??wR?ï???;???Unknown
?'HostReluGrad"+gradient_tape/sequential_1/dense_3/ReluGrad(1      @9      @A      @I      @azu??wR?i?{?%6E???Unknown
?(HostBiasAddGrad"6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGrad(1      @9      @A      @I      @azu??wR?iBnN?qN???Unknown
?)HostReadVariableOp"+sequential_1/dense_4/BiasAdd/ReadVariableOp(1      @9      @A      @I      @azu??wR?i?`?(?W???Unknown
?*HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1      @9      @A      @I      @azu??wR?i?S??`???Unknown
?+HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @azu??wR?isF?+$j???Unknown
V,HostCast"Cast(1      @9      @A      @I      @a70?4??K?i|??q???Unknown
?-HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??a70?4??K?i???m?w???Unknown
`.HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a70?4??K?i????~???Unknown
u/HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a70?4??K?i???օ???Unknown
?0HostReluGrad"+gradient_tape/sequential_1/dense_4/ReluGrad(1      @9      @A      @I      @a70?4??K?i?T?PÌ???Unknown
?1HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a70?4??K?i?????????Unknown
?2HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a70?4??K?i??ߒ?????Unknown
?3HostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @a70?4??K?i???3?????Unknown
?4HostReadVariableOp"+sequential_1/dense_5/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a70?4??K?i?,??u????Unknown
?5HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1      @9      @A      @I      @a70?4??K?i?bvb????Unknown
?6HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      @9      @A      @I      @a70?4??K?i??O????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @azu??wB?iT???????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @azu??wB?i??{??????Unknown
X9HostCast"Cast_2(1       @9       @A       @I       @azu??wB?i/Y(????Unknown
X:HostEqual"Equal(1       @9       @A       @I       @azu??wB?ik~??????Unknown
?;HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      .@9      .@A       @I       @azu??wB?i????c????Unknown
u<HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @azu??wB?i%qI?????Unknown
b=HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @azu??wB?i???[?????Unknown
?>HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @azu??wB?i?c?=????Unknown
??HostReadVariableOp"*sequential_1/dense_4/MatMul/ReadVariableOp(1       @9       @A       @I       @azu??wB?i<?c??????Unknown
?@HostReadVariableOp"*sequential_1/dense_5/MatMul/ReadVariableOp(1       @9       @A       @I       @azu??wB?i?V?x????Unknown
?AHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @azu??wB?i???^????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??azu??w2?i??$?e????Unknown
XCHostCast"Cast_3(1      ??9      ??A      ??I      ??azu??w2?iTI~?????Unknown
XDHostCast"Cast_4(1      ??9      ??A      ??I      ??azu??w2?i??????Unknown
TEHostMul"Mul(1      ??9      ??A      ??I      ??azu??w2?i??1?Q????Unknown
sFHostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??azu??w2?ia???????Unknown
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??azu??w2?i<???????Unknown
yHHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??azu??w2?i??>?>????Unknown
?IHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??azu??w2?in??a?????Unknown
?JHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??azu??w2?ir?A?????Unknown
aKHostIdentity"Identity(1D?l?????9D?l?????AD?l?????ID?l?????a?ol?1?i?????????Unknown?
JLHostReadVariableOp"div_no_nan_1/ReadVariableOp(i?????????Unknown2CPU