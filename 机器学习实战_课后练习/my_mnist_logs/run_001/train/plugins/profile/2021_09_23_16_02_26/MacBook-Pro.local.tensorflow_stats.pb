"?J
BHostIDLE"IDLE1m???ٕ?@Am???ٕ?@a???????i????????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1????K?@9????K?@A????K?@I????K?@a,?ƴ?i??_???Unknown?
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1     ?}@9     ?}@A     ?}@I     ?}@a???O$??i?D]?,????Unknown
sHost_FusedMatMul"sequential_1/dense_3/Relu(1     pw@9     pw@A     pw@I     pw@a??]????i???ݧ????Unknown
wHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      j@9      j@A      j@I      j@a~??^??r?i ??????Unknown
?HostReadVariableOp"*sequential_1/dense_4/MatMul/ReadVariableOp(1     @c@9     @c@A     @c@I     @c@a&?5?Mk?i??y(7???Unknown
iHostWriteSummary"WriteSummary(1=
ףp?b@9=
ףp?b@A=
ףp?b@I=
ףp?b@a.??2??j?i????Q???Unknown?
}HostMatMul")gradient_tape/sequential_1/dense_4/MatMul(1     ?^@9     ?^@A     ?^@I     ?^@a??m?2?e?i3\R?g???Unknown
?	HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      ]@9      ]@A      ]@I      ]@a?V???d?i??i?|???Unknown

HostMatMul"+gradient_tape/sequential_1/dense_4/MatMul_1(1     ?\@9     ?\@A     ?\@I     ?\@a?t?? cd?i?V;?y????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1     @X@9     @X@A     @X@I     @X@a??(?82a?i??.?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a:?_???X?iU?-????Unknown
sHost_FusedMatMul"sequential_1/dense_4/Relu(1     ?L@9     ?L@A     ?L@I     ?L@a???5T?i??x?/????Unknown
^HostGatherV2"GatherV2(1      ;@9      ;@A      ;@I      ;@a?G?/q%C?iq??F?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      9@9      9@A      9@I      9@a)9DR_?A?iF??g????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      <@9      <@A      7@I      7@aY*?tMO@?i?o??{????Unknown
lHostIteratorGetNext"IteratorGetNext(1      6@9      6@A      6@I      6@a?E??3??i3?#b????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      4@9      4@A      4@I      4@aB(mPe]<?i?????????Unknown
HostMatMul"+gradient_tape/sequential_1/dense_5/MatMul_1(1      0@9      0@A      0@I      0@a????6?i?0}??????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_5/MatMul(1      ,@9      ,@A      ,@I      ,@aaϲ??3?iP?R?????Unknown
sHostReadVariableOp"SGD/Cast/ReadVariableOp(1      *@9      *@A      *@I      *@a??A?o2?i?)?O?????Unknown
qHostSoftmax"sequential_1/dense_5/Softmax(1      *@9      *@A      *@I      *@a??A?o2?i@L?L?????Unknown
dHostDataset"Iterator::Model(1     ?B@9     ?B@A      (@I      (@a??tc?1?iֺ???????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?
/?A?)?i??[?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?
/?A?)?i?`??,????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a????&?i?????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a????&?i֞??????Unknown
eHost
LogicalAnd"
LogicalAnd(1j?t?@9j?t?@Aj?t?@Ij?t?@a?oC||?%?icS?b????Unknown?
vHostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aaϲ??#?i:N???????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aaϲ??#?ig9?:?????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aaϲ??#?i?$9?????Unknown
? HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      @9      @A      @I      @aaϲ??#?i?ۙY????Unknown
?!HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aaϲ??#?i??|I?????Unknown
?"HostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @aaϲ??#?i???????Unknown
?#HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a??tc?!?if?F?????Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??tc?!?i?T???????Unknown
?%HostBiasAddGrad"6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??tc?!?i??Q?????Unknown
v&Host_FusedMatMul"sequential_1/dense_5/BiasAdd(1      @9      @A      @I      @a??tc?!?iG÷.????Unknown
g'HostStridedSlice"strided_slice(1      @9      @A      @I      @a??tc?!?i??|&????Unknown
Z(HostArgMax"ArgMax(1      @9      @A      @I      @aB(mPe]?i?}Hg	????Unknown
?)HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @aB(mPe]?idsR?????Unknown
V*HostSum"Sum_2(1      @9      @A      @I      @aB(mPe]?ḯ?=?????Unknown
b+HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aB(mPe]?i6?(?????Unknown
?,HostReadVariableOp"+sequential_1/dense_5/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aB(mPe]?i????????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?????i&[??J????Unknown
u.HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?????i?*?% ????Unknown
`/HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?????i4????????Unknown
?0HostReluGrad"+gradient_tape/sequential_1/dense_3/ReluGrad(1      @9      @A      @I      @a?????i?ɭ7k????Unknown
?1HostReadVariableOp"+sequential_1/dense_4/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?????iB??? ????Unknown
?2HostReadVariableOp"*sequential_1/dense_5/MatMul/ReadVariableOp(1      @9      @A      @I      @a?????i?h?I?????Unknown
v3HostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @a?????iP8zҋ????Unknown
?4HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1      @9      @A      @I      @a?????i?i[A????Unknown
?5HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      @9      @A      @I      @a?????i^?W??????Unknown
?6HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?????i??Fm?????Unknown
t7HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a??tc??i????4????Unknown
|8HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a??tc??i1ެ??????Unknown
?9HostReluGrad"+gradient_tape/sequential_1/dense_4/ReluGrad(1      @9      @A      @I      @a??tc??i??_?D????Unknown
?:HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??tc??i}?????Unknown
?;HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1      @9      @A      @I      @a??tc??i#1?.U????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?????i??=??????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?????i? ??
????Unknown
V>HostCast"Cast(1       @9       @A       @I       @a?????ioh,|e????Unknown
X?HostCast"Cast_2(1       @9       @A       @I       @a?????i3У@?????Unknown
X@HostCast"Cast_3(1       @9       @A       @I       @a?????i?7????Unknown
XAHostCast"Cast_4(1       @9       @A       @I       @a?????i????u????Unknown
XBHostEqual"Equal(1       @9       @A       @I       @a?????i
??????Unknown
uCHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?????iCo?R+????Unknown
?DHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?????i???????Unknown
?EHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a?????i?>p??????Unknown
aFHostIdentity"Identity(1?Zd;??9?Zd;??A?Zd;??I?Zd;??awc瞲?>i?|?????Unknown?
TGHostMul"Mul(1      ??9      ??A      ??I      ??a?????>i|0wJ????Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?????>i^?L?w????Unknown
yIHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?????>i@??;?????Unknown
?JHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?????>i"Lĝ?????Unknown
?KHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?????>i     ???Unknown*?I
uHostFlushSummaryWriter"FlushSummaryWriter(1????K?@9????K?@A????K?@I????K?@a??~?????i??~??????Unknown?
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1     ?}@9     ?}@A     ?}@I     ?}@a?PD\???i?m??-???Unknown
sHost_FusedMatMul"sequential_1/dense_3/Relu(1     pw@9     pw@A     pw@I     pw@a?;J?{??i???t%???Unknown
wHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      j@9      j@A      j@I      j@a?l5狡?i?v? 3>???Unknown
?HostReadVariableOp"*sequential_1/dense_4/MatMul/ReadVariableOp(1     @c@9     @c@A     @c@I     @c@a?O&a?ۙ?i	??K???Unknown
iHostWriteSummary"WriteSummary(1=
ףp?b@9=
ףp?b@A=
ףp?b@I=
ףp?b@a??*ŁL??i?? Zt????Unknown?
}HostMatMul")gradient_tape/sequential_1/dense_4/MatMul(1     ?^@9     ?^@A     ?^@I     ?^@a?嫯???i?\?O?|???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      ]@9      ]@A      ]@I      ]@aa??:z??i?9> ???Unknown
	HostMatMul"+gradient_tape/sequential_1/dense_4/MatMul_1(1     ?\@9     ?\@A     ?\@I     ?\@a۽?=O??i?(,?????Unknown
?
HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1     @X@9     @X@A     @X@I     @X@a?Bfm?I??i?Z?3E5???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?Q@9     ?Q@A     ?Q@I     ?Q@a???)܁??i?q>?L????Unknown
sHost_FusedMatMul"sequential_1/dense_4/Relu(1     ?L@9     ?L@A     ?L@I     ?L@a???A$??irܫ?????Unknown
^HostGatherV2"GatherV2(1      ;@9      ;@A      ;@I      ;@a??bY"r?i6??^"???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      9@9      9@A      9@I      9@aφ?x?p?iD? P?%???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      <@9      <@A      7@I      7@a?b?/?n?i</??D???Unknown
lHostIteratorGetNext"IteratorGetNext(1      6@9      6@A      6@I      6@aD?tJO?m?i)?G?)b???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      4@9      4@A      4@I      4@a?ט???j?i=.]}???Unknown
HostMatMul"+gradient_tape/sequential_1/dense_5/MatMul_1(1      0@9      0@A      0@I      0@a???~e?i?Mh?????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_5/MatMul(1      ,@9      ,@A      ,@I      ,@a???I?b?iE"?S????Unknown
sHostReadVariableOp"SGD/Cast/ReadVariableOp(1      *@9      *@A      *@I      *@a4?	iva?i?8ʶ???Unknown
qHostSoftmax"sequential_1/dense_5/Softmax(1      *@9      *@A      *@I      *@a4?	iva?i]O?@????Unknown
dHostDataset"Iterator::Model(1     ?B@9     ?B@A      (@I      (@ak?(W?`?i?wq_????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a ¼??-X?i?ֲ?u????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a ¼??-X?i?4?،????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a???~U?i????K????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a???~U?iL?
???Unknown
eHost
LogicalAnd"
LogicalAnd(1j?t?@9j?t?@Aj?t?@Ij?t?@a?D???T?iQu???Unknown?
vHostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???I?R?i9uv????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???I?R?i??R?C#???Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???I?R?i?0??,???Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      @9      @A      @I      @a???I?R?i??6???Unknown
? HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???I?R?ie?	y????Unknown
?!HostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @a???I?R?i???.?H???Unknown
?"HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @ak?(W?P?i?&?r?P???Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @ak?(W?P?i2???X???Unknown
?$HostBiasAddGrad"6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ak?(W?P?isOK?a???Unknown
v%Host_FusedMatMul"sequential_1/dense_5/BiasAdd(1      @9      @A      @I      @ak?(W?P?i??v?i???Unknown
g&HostStridedSlice"strided_slice(1      @9      @A      @I      @ak?(W?P?i?w??,q???Unknown
Z'HostArgMax"ArgMax(1      @9      @A      @I      @a?ט???J?i+??w???Unknown
?(HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a?ט???J?iaĕJ?~???Unknown
V)HostSum"Sum_2(1      @9      @A      @I      @a?ט???J?i?j?R????Unknown
b*HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?ט???J?i??
????Unknown
?+HostReadVariableOp"+sequential_1/dense_5/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?ט???J?i?u?????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a???~E?i.o?? ????Unknown
u-HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a???~E?iY'?z?????Unknown
`.HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a???~E?i??Y?ߢ???Unknown
?/HostReluGrad"+gradient_tape/sequential_1/dense_3/ReluGrad(1      @9      @A      @I      @a???~E?i??!??????Unknown
?0HostReadVariableOp"+sequential_1/dense_4/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???~E?i?O??????Unknown
?1HostReadVariableOp"*sequential_1/dense_5/MatMul/ReadVariableOp(1      @9      @A      @I      @a???~E?i???????Unknown
v2HostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @a???~E?i0?x^????Unknown
?3HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1      @9      @A      @I      @a???~E?i[x@??????Unknown
?4HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      @9      @A      @I      @a???~E?i?0????Unknown
?5HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a???~E?i??ϐ|????Unknown
t6HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @ak?(W?@?iѲ?2?????Unknown
|7HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @ak?(W?@?i?|?ԋ????Unknown
?8HostReluGrad"+gradient_tape/sequential_1/dense_4/ReluGrad(1      @9      @A      @I      @ak?(W?@?iGw?????Unknown
?9HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @ak?(W?@?i1'?????Unknown
?:HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1      @9      @A      @I      @ak?(W?@?iQ?<??????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a???~5?ig??|R????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a???~5?i}?>????Unknown
V=HostCast"Cast(1       @9       @A       @I       @a???~5?i?oh??????Unknown
X>HostCast"Cast_2(1       @9       @A       @I       @a???~5?i?K??a????Unknown
X?HostCast"Cast_3(1       @9       @A       @I       @a???~5?i?'0?????Unknown
X@HostCast"Cast_4(1       @9       @A       @I       @a???~5?i??C?????Unknown
XAHostEqual"Equal(1       @9       @A       @I       @a???~5?i???q????Unknown
uBHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a???~5?i?[? ????Unknown
?CHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a???~5?i????????Unknown
?DHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a???~5?i-t#I?????Unknown
aEHostIdentity"Identity(1?Zd;??9?Zd;??A?Zd;??I?Zd;??a*?Y.6?,?i?Y??H????Unknown?
TFHostMul"Mul(1      ??9      ??A      ??I      ??a???~%?i?G8}?????Unknown
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???~%?i?5?]?????Unknown
yHHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a???~%?i?#?>P????Unknown
?IHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a???~%?i?N?????Unknown
?JHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a???~%?i      ???Unknown2CPU