"?C
BHostIDLE"IDLE1     ??@A     ??@a?\????i?\?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     X?@9     X?@A     X?@I     X?@a?o??o???i???Unknown?
?HostReadVariableOp",sequential_7/dense_22/BiasAdd/ReadVariableOp(1     `k@9     `k@A     `k@I     `k@aMrMr??i1?z1?????Unknown
iHostWriteSummary"WriteSummary(1      C@9      C@A      C@I      C@a??o??ot?i?~??~#???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1     ?@@9     ?@@A     ?@@I     ?@@aRv?Qv?q?i??F??F???Unknown
THostMul"Mul(1      ;@9      ;@A      ;@I      ;@a??
??
m?i???d???Unknown
tHost_FusedMatMul"sequential_7/dense_21/Relu(1      7@9      7@A      7@I      7@a?O??O?h?i??|??|???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      4@9      4@A      4@I      4@a:?:?e?i%I?$I????Unknown
l	HostIteratorGetNext"IteratorGetNext(1      4@9      4@A      4@I      4@a:?:?e?i_̧^̧???Unknown
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      4@9      4@A      3@I      3@a??o??od?i=<<<<????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_21/MatMul(1      3@9      3@A      3@I      3@a??o??od?i???????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_22/MatMul(1      2@9      2@A      2@I      2@a?\?\c?i???????Unknown
?HostMatMul",gradient_tape/sequential_7/dense_22/MatMul_1(1      .@9      .@A      .@I      .@a?k"?k"`?i+t+????Unknown
uHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ,@9      ,@A      ,@I      ,@a^?i:?:???Unknown
tHost_FusedMatMul"sequential_7/dense_22/Relu(1      (@9      (@A      (@I      (@a????Y?im"?k"???Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_23/MatMul(1      &@9      &@A      &@I      &@a????W?ig?e????Unknown
wHostMul"&gradient_tape/mean_squared_error/mul_1(1      $@9      $@A      $@I      $@a:?:?U?i?&?&???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?\?\S?iEg?Bg0???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?\?\S?i?:?:???Unknown
dHostDataset"Iterator::Model(1     ?D@9     ?D@A       @I       @a?5?5Q?ij?Bg?B???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a?5?5Q?iNKKKKK???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a?5?5Q?i2?S/?S???Unknown
?HostMatMul",gradient_tape/sequential_7/dense_23/MatMul_1(1       @9       @A       @I       @a?5?5Q?i?\?\???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?5?5Q?i?e?e???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @aN?i???~?l???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aN?i
+t+t???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aN?i?????{???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a????I?i?&?&????Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_22/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????I?i??䚈???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_23/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????I?i????Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a????I?i>?:?????Unknown
t HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a:?:?E?i???????Unknown
`!HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a:?:?E?i?D ?D????Unknown
?"HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a:?:?E?i?????????Unknown
?#HostBiasAddGrad"7gradient_tape/sequential_7/dense_21/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a:?:?E?iz+t????Unknown
?$HostReadVariableOp"+sequential_7/dense_22/MatMul/ReadVariableOp(1      @9      @A      @I      @a:?:?E?iIg?Bg????Unknown
?%HostReadVariableOp",sequential_7/dense_23/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a:?:?E?i?5ȵ???Unknown
V&HostCast"Cast(1      @9      @A      @I      @a?5?5A?i?:?????Unknown
?'HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?5?5A?i?b>?b????Unknown
w(Host_FusedMatMul"sequential_7/dense_23/BiasAdd(1      @9      @A      @I      @a?5?5A?in?Bg?????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a????9?i???|?????Unknown
?*HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a????9?i?$I?$????Unknown
u+HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a????9?i?^̧^????Unknown
},HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @a????9?iƘO??????Unknown
?-HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a????9?i?????????Unknown
?.HostReadVariableOp",sequential_7/dense_21/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????9?i?V?????Unknown
?/HostReadVariableOp"+sequential_7/dense_21/MatMul/ReadVariableOp(1      @9      @A      @I      @a????9?iG??F????Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?5?51?i?m۶m????Unknown
s1HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?5?51?iz??o?????Unknown
`2HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?5?51?i3??(?????Unknown
?3HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a?5?51?i?????????Unknown
w4HostCast"%gradient_tape/mean_squared_error/Cast(1       @9       @A       @I       @a?5?51?i???????Unknown
u5HostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @a?5?51?i^/?S/????Unknown
6HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1       @9       @A       @I       @a?5?51?iV?V????Unknown
u7HostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a?5?51?i?|??|????Unknown
}8HostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a?5?51?i???~?????Unknown
?9HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?5?51?iB??7?????Unknown
?:HostReluGrad",gradient_tape/sequential_7/dense_21/ReluGrad(1       @9       @A       @I       @a?5?51?i?????????Unknown
?;HostReluGrad",gradient_tape/sequential_7/dense_22/ReluGrad(1       @9       @A       @I       @a?5?51?i???????Unknown
i<HostMean"mean_squared_error/Mean(1       @9       @A       @I       @a?5?51?im>?b>????Unknown
?=HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?5?51?i&e?e????Unknown
?>HostReadVariableOp"+sequential_7/dense_23/MatMul/ReadVariableOp(1       @9       @A       @I       @a?5?51?iߋ?ԋ????Unknown
a?HostIdentity"Identity(1      ??9      ??A      ??I      ??a?5?5!?i<?z1?????Unknown?
?@HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a?5?5!?i?????????Unknown
|AHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?5?5!?i??|??????Unknown
wBHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?5?5!?iS??F?????Unknown
uCHostMul"$gradient_tape/mean_squared_error/Mul(1      ??9      ??A      ??I      ??a?5?5!?i??~??????Unknown
|DHostDivNoNan"&mean_squared_error/weighted_loss/value(1      ??9      ??A      ??I      ??a?5?5!?i     ???Unknown*?B
uHostFlushSummaryWriter"FlushSummaryWriter(1     X?@9     X?@A     X?@I     X?@aI3?gg^??iI3?gg^???Unknown?
?HostReadVariableOp",sequential_7/dense_22/BiasAdd/ReadVariableOp(1     `k@9     `k@A     `k@I     `k@a???!????i?H< j???Unknown
iHostWriteSummary"WriteSummary(1      C@9      C@A      C@I      C@amQ?џ?imc?!?h???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1     ?@@9     ?@@A     ?@@I     ?@@a?rMc???i??9?E???Unknown
THostMul"Mul(1      ;@9      ;@A      ;@I      ;@a?F?Q???i:????????Unknown
tHost_FusedMatMul"sequential_7/dense_21/Relu(1      7@9      7@A      7@I      7@a?~E??A??i/7??????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      4@9      4@A      4@I      4@a???澐?is??t????Unknown
lHostIteratorGetNext"IteratorGetNext(1      4@9      4@A      4@I      4@a???澐?i????????Unknown
?	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      4@9      4@A      3@I      3@amQ?я?ikQ?????Unknown
~
HostMatMul"*gradient_tape/sequential_7/dense_21/MatMul(1      3@9      3@A      3@I      3@amQ?я?i?b?????Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_22/MatMul(1      2@9      2@A      2@I      2@a 	?l$??iC?l?????Unknown
?HostMatMul",gradient_tape/sequential_7/dense_22/MatMul_1(1      .@9      .@A      .@I      .@a??Z??i?u? |???Unknown
uHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ,@9      ,@A      ,@I      ,@a?xIW?q??i?(?L?????Unknown
tHost_FusedMatMul"sequential_7/dense_22/Relu(1      (@9      (@A      (@I      (@a???H??i\??lH*???Unknown
~HostMatMul"*gradient_tape/sequential_7/dense_23/MatMul(1      &@9      &@A      &@I      &@a?L?V?k??i?T4??s???Unknown
wHostMul"&gradient_tape/mean_squared_error/mul_1(1      $@9      $@A      $@I      $@a???澀?i0??d?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a 	?l$~?iB??<;????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a 	?l$~?iT???/???Unknown
dHostDataset"Iterator::Model(1     ?D@9     ?D@A       @I       @a?@??
?z?i?lH*e???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a?@??
?z?iX9???????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a?@??
?z?i??TF????Unknown
?HostMatMul",gradient_tape/sequential_7/dense_23/MatMul_1(1       @9       @A       @I       @a?@??
?z?i\?Uj????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?@??
?z?iޞ?r;???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?xIW?qw?i?1^?Uj???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?xIW?qw?i??%9????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?xIW?qw?i?W?w????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a???Ht?i??M????Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_22/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???Ht?iv
}???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_7/dense_23/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???Ht?i?c?'?@???Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a???Ht?i8?ȷ?h???Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?????p?i	? ?\????Unknown
` HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?????p?i??xRګ???Unknown
?!HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a?????p?i??X????Unknown
?"HostBiasAddGrad"7gradient_tape/sequential_7/dense_21/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?????p?i|<)??????Unknown
?#HostReadVariableOp"+sequential_7/dense_22/MatMul/ReadVariableOp(1      @9      @A      @I      @a?????p?iM\??S???Unknown
?$HostReadVariableOp",sequential_7/dense_23/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?????p?i|ه?1???Unknown
V%HostCast"Cast(1      @9      @A      @I      @a?@??
?j?i_b???L???Unknown
?&HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?@??
?j?i?H3?gg???Unknown
w'Host_FusedMatMul"sequential_7/dense_23/BiasAdd(1      @9      @A      @I      @a?@??
?j?i?.??2????Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a???Hd?i????J????Unknown
?)HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a???Hd?iC??7c????Unknown
u*HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a???Hd?i?4?{????Unknown
}+HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @a???Hd?i???Ǔ????Unknown
?,HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a???Hd?iV???????Unknown
?-HostReadVariableOp",sequential_7/dense_21/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???Hd?i;?W?????Unknown
?.HostReadVariableOp"+sequential_7/dense_21/MatMul/ReadVariableOp(1      @9      @A      @I      @a???Hd?i????????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?@??
?Z?i?ZB%B???Unknown
s0HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?@??
?Z?i?͘??)???Unknown
`1HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?@??
?Z?iA?/7???Unknown
?2HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a?@??
?Z?i8?E?rD???Unknown
w3HostCast"%gradient_tape/mean_squared_error/Cast(1       @9       @A       @I       @a?@??
?Z?iX'?:?Q???Unknown
u4HostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @a?@??
?Z?ix???=_???Unknown
5HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1       @9       @A       @I       @a?@??
?Z?i?IE?l???Unknown
u6HostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a?@??
?Z?i????z???Unknown
}7HostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a?@??
?Z?i???On????Unknown
?8HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?@??
?Z?i?fL?Ӕ???Unknown
?9HostReluGrad",gradient_tape/sequential_7/dense_21/ReluGrad(1       @9       @A       @I       @a?@??
?Z?iڢZ9????Unknown
?:HostReluGrad",gradient_tape/sequential_7/dense_22/ReluGrad(1       @9       @A       @I       @a?@??
?Z?i8M?ߞ????Unknown
i;HostMean"mean_squared_error/Mean(1       @9       @A       @I       @a?@??
?Z?iX?Oe????Unknown
?<HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?@??
?Z?ix3??i????Unknown
?=HostReadVariableOp"+sequential_7/dense_23/MatMul/ReadVariableOp(1       @9       @A       @I       @a?@??
?Z?i???o?????Unknown
a>HostIdentity"Identity(1      ??9      ??A      ??I      ??a?@??
?J?i(??2?????Unknown?
??HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a?@??
?J?i?S?4????Unknown
|@HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?@??
?J?iHS???????Unknown
wAHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?@??
?J?i،?z?????Unknown
uBHostMul"$gradient_tape/mean_squared_error/Mul(1      ??9      ??A      ??I      ??a?@??
?J?ih?T=M????Unknown
|CHostDivNoNan"&mean_squared_error/weighted_loss/value(1      ??9      ??A      ??I      ??a?@??
?J?i?????????Unknown2CPU