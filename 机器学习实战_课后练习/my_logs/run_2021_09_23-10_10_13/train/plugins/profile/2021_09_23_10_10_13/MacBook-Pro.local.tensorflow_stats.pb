"?C
BHostIDLE"IDLE1     ??@A     ??@a?i:????i?i:?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@aX?W~4??i+c?hZ???Unknown?
iHostWriteSummary"WriteSummary(1      =@9      =@A      =@I      =@a? C*??o?i,?H]*z???Unknown?
tHost_FusedMatMul"sequential_5/dense_15/Relu(1      6@9      6@A      6@I      6@a???Eh?i:N?A????Unknown
~HostMatMul"*gradient_tape/sequential_5/dense_15/MatMul(1      5@9      5@A      5@I      5@a|5n??f?iPG??@????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      1@9      1@A      1@I      1@a?O???b?i?9?0޻???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      1@I      1@a?O???b?i?+??{????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      .@9      .@A      .@I      .@a????l`?i????????Unknown
~	HostMatMul"*gradient_tape/sequential_5/dense_16/MatMul(1      .@9      .@A      .@I      .@a????l`?i????U????Unknown
l
HostIteratorGetNext"IteratorGetNext(1      (@9      (@A      (@I      (@a?a?4?GZ?i[F??y????Unknown
?HostMatMul",gradient_tape/sequential_5/dense_16/MatMul_1(1      &@9      &@A      &@I      &@a???EX?iR;R????Unknown
`HostGatherV2"
GatherV2_1(1      "@9      "@A      "@I      "@aF	y???S?i?̎K`???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aF	y???S?i\??D;???Unknown
~HostMatMul"*gradient_tape/sequential_5/dense_17/MatMul(1      "@9      "@A      "@I      "@aF	y???S?i?E6>&???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @aw?kxI?Q?i?{???.???Unknown
dHostDataset"Iterator::Model(1      9@9      9@A       @I       @aw?kxI?Q?iw????7???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @aw?kxI?Q?iB?j,^@???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @aw?kxI?Q?i'? I???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aQG??@?N?i?K!?P???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_5/dense_17/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aQG??@?N?i1{pquX???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aQG??@?N?iC*??`???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?a?4?GJ?i?R"??f???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?a?4?GJ?i?z??Cm???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?a?4?GJ?iK?<??s???Unknown
?HostMatMul",gradient_tape/sequential_5/dense_17/MatMul_1(1      @9      @A      @I      @a?a?4?GJ?i??ɯgz???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a|?֛?E?iBm?V????Unknown
?HostBiasAddGrad"7gradient_tape/sequential_5/dense_16/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a|?֛?E?i???Z????Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a|?֛?E?i????Ԋ???Unknown
tHost_FusedMatMul"sequential_5/dense_16/Relu(1      @9      @A      @I      @a|?֛?E?iR?KN????Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aw?kxI?A?im???????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @aw?kxI?A?i??\?????Unknown
u HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aw?kxI?A?iѢ?Br????Unknown
?!HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aw?kxI?A?i???ӡ???Unknown
w"HostCast"%gradient_tape/mean_squared_error/Cast(1      @9      @A      @I      @aw?kxI?A?i??v?4????Unknown
?#HostBiasAddGrad"7gradient_tape/sequential_5/dense_15/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aw?kxI?A?i???9?????Unknown
i$HostMean"mean_squared_error/Mean(1      @9      @A      @I      @aw?kxI?A?ii3??????Unknown
?%HostReadVariableOp",sequential_5/dense_15/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aw?kxI?A?iO)??X????Unknown
?&HostReadVariableOp"+sequential_5/dense_15/MatMul/ReadVariableOp(1      @9      @A      @I      @aw?kxI?A?i5D?0?????Unknown
?'HostReadVariableOp",sequential_5/dense_16/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aw?kxI?A?i_M?????Unknown
?(HostReadVariableOp"+sequential_5/dense_16/MatMul/ReadVariableOp(1      @9      @A      @I      @aw?kxI?A?iz??|????Unknown
w)Host_FusedMatMul"sequential_5/dense_17/BiasAdd(1      @9      @A      @I      @aw?kxI?A?i??	(?????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?a?4?G:?i)?%'????Unknown
V+HostCast"Cast(1      @9      @A      @I      @a?a?4?G:?i???#p????Unknown
},HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @a?a?4?G:?ikQ]!?????Unknown
u-HostSub"$gradient_tape/mean_squared_error/sub(1      @9      @A      @I      @a?a?4?G:?i??#????Unknown
?.HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a?a?4?G:?i?y?K????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @aw?kxI?1?i6??{????Unknown
T0HostMul"Mul(1       @9       @A       @I       @aw?kxI?1?i??Ho?????Unknown
s1HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @aw?kxI?1?i?w?????Unknown
|2HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aw?kxI?1?i????????Unknown
u3HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aw?kxI?1?i??j>????Unknown
u4HostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @aw?kxI?1?iu?o????Unknown
u5HostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @aw?kxI?1?i??3??????Unknown
6HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1       @9       @A       @I       @aw?kxI?1?i[?bf?????Unknown
w7HostMul"&gradient_tape/mean_squared_error/mul_1(1       @9       @A       @I       @aw?kxI?1?i???????Unknown
?8HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aw?kxI?1?iA ??1????Unknown
?9HostReluGrad",gradient_tape/sequential_5/dense_15/ReluGrad(1       @9       @A       @I       @aw?kxI?1?i??ab????Unknown
?:HostReluGrad",gradient_tape/sequential_5/dense_16/ReluGrad(1       @9       @A       @I       @aw?kxI?1?i'?????Unknown
?;HostSquaredDifference"$mean_squared_error/SquaredDifference(1       @9       @A       @I       @aw?kxI?1?i?(N??????Unknown
?<HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aw?kxI?1?i6}]?????Unknown
|=HostDivNoNan"&mean_squared_error/weighted_loss/value(1       @9       @A       @I       @aw?kxI?1?i?C?%????Unknown
?>HostReadVariableOp",sequential_5/dense_17/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aw?kxI?1?i?PۯU????Unknown
??HostReadVariableOp"+sequential_5/dense_17/MatMul/ReadVariableOp(1       @9       @A       @I       @aw?kxI?1?if^
Y?????Unknown
a@HostIdentity"Identity(1      ??9      ??A      ??I      ??aw?kxI?!?i塭?????Unknown?
`AHostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??aw?kxI?!?i?k9?????Unknown
wBHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aw?kxI?!?i???V?????Unknown
?CHostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??aw?kxI?!?iJyh??????Unknown
}DHostRealDiv"(gradient_tape/mean_squared_error/truediv(1      ??9      ??A      ??I      ??aw?kxI?!?i     ???Unknown*?B
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@as?U????is?U?????Unknown?
iHostWriteSummary"WriteSummary(1      =@9      =@A      =@I      =@a??V"1??i??d{?????Unknown?
tHost_FusedMatMul"sequential_5/dense_15/Relu(1      6@9      6@A      6@I      6@aY}?E+???i????7X???Unknown
~HostMatMul"*gradient_tape/sequential_5/dense_15/MatMul(1      5@9      5@A      5@I      5@aձ?6Ls??i]?F7????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      1@9      1@A      1@I      1@aŃ??????i{?"??????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      1@I      1@aŃ??????i?)?6?C???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      .@9      .@A      .@I      .@a???????i???ŏ????Unknown
~HostMatMul"*gradient_tape/sequential_5/dense_16/MatMul(1      .@9      .@A      .@I      .@a???????ie8?T?O???Unknown
l	HostIteratorGetNext"IteratorGetNext(1      (@9      (@A      (@I      (@aac?̊?i??j?ú???Unknown
?
HostMatMul",gradient_tape/sequential_5/dense_16/MatMul_1(1      &@9      &@A      &@I      &@aY}?E+???i???????Unknown
`HostGatherV2"
GatherV2_1(1      "@9      "@A      "@I      "@aIOT
???i??com???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aIOT
???i&U?ֽ???Unknown
~HostMatMul"*gradient_tape/sequential_5/dense_17/MatMul(1      "@9      "@A      "@I      "@aIOT
???ic???<???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @aA????݁?iD????U???Unknown
dHostDataset"Iterator::Model(1      9@9      9@A       @I       @aA????݁?i%?bc,????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @aA????݁?i'?????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @aA????݁?i?q??,???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aqBJ?eD?il??j???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_5/dense_17/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aqBJ?eD?i??A?-????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aqBJ?eD?iv/~L?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aac??z?i?eDP???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aac??z?iț
??R???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aac??z?i???ă????Unknown
?HostMatMul",gradient_tape/sequential_5/dense_17/MatMul_1(1      @9      @A      @I      @aac??z?i??????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @aQ??'mUv?i???q?????Unknown
?HostBiasAddGrad"7gradient_tape/sequential_5/dense_16/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aQ??'mUv?i??6Ls???Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @aQ??'mUv?i???&D???Unknown
tHost_FusedMatMul"sequential_5/dense_16/Relu(1      @9      @A      @I      @aQ??'mUv?iNg? ?p???Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aA?????q?i???℔???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @aA?????q?i0Z??@????Unknown
uHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aA?????q?i??b??????Unknown
? HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aA?????q?iM<??????Unknown
w!HostCast"%gradient_tape/mean_squared_error/Cast(1      @9      @A      @I      @aA?????q?i??jt#???Unknown
?"HostBiasAddGrad"7gradient_tape/sequential_5/dense_15/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aA?????q?i???K0G???Unknown
i#HostMean"mean_squared_error/Mean(1      @9      @A      @I      @aA?????q?ie??-?j???Unknown
?$HostReadVariableOp",sequential_5/dense_15/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aA?????q?i?2??????Unknown
?%HostReadVariableOp"+sequential_5/dense_15/MatMul/ReadVariableOp(1      @9      @A      @I      @aA?????q?iG?{?c????Unknown
?&HostReadVariableOp",sequential_5/dense_16/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aA?????q?i?%U?????Unknown
?'HostReadVariableOp"+sequential_5/dense_16/MatMul/ReadVariableOp(1      @9      @A      @I      @aA?????q?i)?.??????Unknown
w(Host_FusedMatMul"sequential_5/dense_17/BiasAdd(1      @9      @A      @I      @aA?????q?i??????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aac??j?i?3k?d8???Unknown
V*HostCast"Cast(1      @9      @A      @I      @aac??j?i?N?i1S???Unknown
}+HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @aac??j?i?i1S?m???Unknown
u,HostSub"$gradient_tape/mean_squared_error/sub(1      @9      @A      @I      @aac??j?iꄔ<ˈ???Unknown
?-HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @aac??j?i???%?????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @aA?????a?i?\?v????Unknown
T/HostMul"Mul(1       @9       @A       @I       @aA?????a?in?T????Unknown
s0HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @aA?????a?i&ֽ?1????Unknown
|1HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aA?????a?iޒ??????Unknown
u2HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aA?????a?i?O???????Unknown
u3HostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @aA?????a?iN??????Unknown
u4HostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @aA?????a?i?p?? ???Unknown
5HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1       @9       @A       @I       @aA?????a?i??]??2???Unknown
w6HostMul"&gradient_tape/mean_squared_error/mul_1(1       @9       @A       @I       @aA?????a?ivBJ?eD???Unknown
?7HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aA?????a?i.?6?CV???Unknown
?8HostReluGrad",gradient_tape/sequential_5/dense_15/ReluGrad(1       @9       @A       @I       @aA?????a?i??#?!h???Unknown
?9HostReluGrad",gradient_tape/sequential_5/dense_16/ReluGrad(1       @9       @A       @I       @aA?????a?i?xq?y???Unknown
?:HostSquaredDifference"$mean_squared_error/SquaredDifference(1       @9       @A       @I       @aA?????a?iV5?a݋???Unknown
?;HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aA?????a?i??R?????Unknown
|<HostDivNoNan"&mean_squared_error/weighted_loss/value(1       @9       @A       @I       @aA?????a?iƮ?C?????Unknown
?=HostReadVariableOp",sequential_5/dense_17/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aA?????a?i~k?4w????Unknown
?>HostReadVariableOp"+sequential_5/dense_17/MatMul/ReadVariableOp(1       @9       @A       @I       @aA?????a?i6(?%U????Unknown
a?HostIdentity"Identity(1      ??9      ??A      ??I      ??aA?????Q?i??&D????Unknown?
`@HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??aA?????Q?i???3????Unknown
wAHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aA?????Q?iJC"????Unknown
?BHostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??aA?????Q?i???????Unknown
}CHostRealDiv"(gradient_tape/mean_squared_error/truediv(1      ??9      ??A      ??I      ??aA?????Q?i     ???Unknown2CPU