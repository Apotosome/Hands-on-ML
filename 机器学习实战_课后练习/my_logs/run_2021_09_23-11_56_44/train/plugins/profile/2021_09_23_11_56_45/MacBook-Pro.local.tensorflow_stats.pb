"?A
BHostIDLE"IDLE1     ??@A     ??@a??<<^??i??<<^???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      P@9      P@A      P@I      P@a?\?6?d??i???b1???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?M@9     ?M@A     ?M@I     ?M@aZ?G??T??iOIcO
????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      ?@9      ?@A      ?@I      ?@a?	֌????iv??	QZ???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      2@I      2@a(?[?o?}?i?XQ鳕???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      2@9      2@A      2@I      2@a(?[?o?}?i?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      0@9      0@A      0@I      0@a?\?6?dz?i?$ys????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a??R%r?i#?+*???Unknown?
}	HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      &@9      &@A      &@I      &@a??R%r?iM!ϽuN???Unknown
l
HostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a?yBp?iANS?so???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?\?6?dj?i?؉?؉???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a?\?6?dj?i?b?r=????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1       @9       @A       @I       @a?\?6?dj?iX??G?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1       @9       @A       @I       @a?\?6?dj?i?w-????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?\?6?dj?id?k????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??:g?i#??,?
???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @ap?????c?i????O???Unknown
dHostDataset"Iterator::Model(1     ?Q@9     ?Q@A      @I      @ap?????c?i???l2???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @ap?????c?irr??E???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?????c?i7Z???Y???Unknown
wHostCast"%gradient_tape/mean_squared_error/Cast(1      @9      @A      @I      @ap?????c?i?A?L~m???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @ap?????c?i?)i?I????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ap?????c?i?R?????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @ap?????c?iK?:,?????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?yB`?i?}1`????Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a?yB`?i?&?6?????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      @9      @A      @I      @a?yB`?i?<<^????Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?\?6?dZ?i灜??????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?\?6?dZ?i?7?????Unknown?
sHostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?\?6?dZ?iC?{????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?\?6?dZ?iqQn?'???Unknown
? HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?\?6?dZ?i??	QZ???Unknown
?!HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?\?6?dZ?i?ۤ??)???Unknown
?"HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??ap?????S?i?O??r3???Unknown
?#HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?????S?i?Í[X=???Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?????S?iv7?+>G???Unknown
u%HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @ap?????S?iY?v?#Q???Unknown
?&HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @ap?????S?i<k?	[???Unknown
?'HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @ap?????S?i?_??d???Unknown
t(Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @ap?????S?iTk?n???Unknown
?)HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @ap?????S?i?zH;?x???Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?\?6?dJ?i|?pT???Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_2(1       @9       @A       @I       @a?\?6?dJ?i????????Unknown
V,HostCast"Cast(1       @9       @A       @I       @a?\?6?dJ?i?b1ۆ????Unknown
T-HostMul"Mul(1       @9       @A       @I       @a?\?6?dJ?iA ????Unknown
|.HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?\?6?dJ?iا?E?????Unknown
?/HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a?\?6?dJ?ioJ{R????Unknown
}0HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a?\?6?dJ?i?g??????Unknown
u1HostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a?\?6?dJ?i???儭???Unknown
}2HostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a?\?6?dJ?i42????Unknown
?3HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?\?6?dJ?i??PP?????Unknown
}4HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?\?6?dJ?ibw??P????Unknown
5HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a?\?6?dJ?i????????Unknown
i6HostMean"mean_squared_error/Mean(1       @9       @A       @I       @a?\?6?dJ?i??9??????Unknown
?7HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?\?6?dJ?i'_?%????Unknown
?8HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a?\?6?dJ?i??Z?????Unknown
?9HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a?\?6?dJ?iU?"?N????Unknown
u:HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?\?6?d:?i?u?*?????Unknown
`;HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a?\?6?d:?i?Fp??????Unknown
w<HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?\?6?d:?i9`4????Unknown
u=HostMul"$gradient_tape/mean_squared_error/Mul(1      ??9      ??A      ??I      ??a?\?6?d:?i?????????Unknown
u>HostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??a?\?6?d:?iѺd??????Unknown
?HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      ??9      ??A      ??I      ??a?\?6?d:?i?0????Unknown
|@HostDivNoNan"&mean_squared_error/weighted_loss/value(1      ??9      ??A      ??I      ??a?\?6?d:?ii]??f????Unknown
?AHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?\?6?d:?i?.Ye?????Unknown
?BHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?\?6?d:?i      ???Unknown
4CHostIdentity"Identity(i      ???Unknown?
JDHostMul"&gradient_tape/mean_squared_error/mul_1(i      ???Unknown*?A
sHostDataset"Iterator::Model::ParallelMapV2(1      P@9      P@A      P@I      P@a7Z?????i7Z??????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?M@9     ?M@A     ?M@I     ?M@a?*;H??i???΃???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      ?@9      ?@A      ?@I      ?@aXe??)??i@?D?č???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      2@I      2@a?}%????i?9IW?0???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      2@9      2@A      2@I      2@a?}%????i??Moz????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      0@9      0@A      0@I      0@a7Z?????i?0?Y+???Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a??#?ƙ?iY?T?????Unknown?
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      &@9      &@A      &@I      &@a??#?ƙ?i
Y?B????Unknown
l	HostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a?????n??i1?W?m???Unknown
^
HostGatherV2"GatherV2(1       @9       @A       @I       @a7Z?????i???΃???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a7Z?????i??=?{????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1       @9       @A       @I       @a7Z?????i\T?gs/???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1       @9       @A       @I       @a7Z?????i&#4k????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a7Z?????i??? c[???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a/??\g??iOozӛ????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?R?Uv??i??ЬO???Unknown
dHostDataset"Iterator::Model(1     ?Q@9     ?Q@A      @I      @a?R?Uv??i??&??????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?R?Uv??i-?|_	0???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?R?Uv??iw??8?????Unknown
wHostCast"%gradient_tape/mean_squared_error/Cast(1      @9      @A      @I      @a?R?Uv??i?)????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a?R?Uv??i?v????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?R?Uv??iU<???????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a?R?Uv??i?Y+?jb???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?????n??i??}%????Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a?????n??i?ߺ]????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      @9      @A      @I      @a?????n??iآ?=?{???Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a7Z?????i??#?????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a7Z?????i?t?	????Unknown?
sHostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a7Z?????il?.??\???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a7Z?????iHFh֊????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a7Z?????i$????????Unknown
? HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a7Z?????i ۢ?=???Unknown
?!HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??a?R?Uv|?i?&???u???Unknown
?"HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?R?Uv|?iJ51|?????Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?R?Uv|?i?C?h9????Unknown
u$HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a?R?Uv|?i?R?Uv???Unknown
?%HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a?R?Uv|?i9a2B?V???Unknown
?&HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a?R?Uv|?i?o?.?????Unknown
t'Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a?R?Uv|?i?~?-????Unknown
?(HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?R?Uv|?i(?3j????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a7Z???r?i?AP??$???Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1       @9       @A       @I       @a7Z???r?i?l?eJ???Unknown
V+HostCast"Cast(1       @9       @A       @I       @a7Z???r?ir????o???Unknown
T,HostMul"Mul(1       @9       @A       @I       @a7Z???r?i?^??a????Unknown
|-HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a7Z???r?iN??ߺ???Unknown
?.HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a7Z???r?i??ߺ]????Unknown
}/HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a7Z???r?i*|??????Unknown
u0HostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a7Z???r?i?0?Y+???Unknown
}1HostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a7Z???r?i?5??P???Unknown
?2HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a7Z???r?it?R?Uv???Unknown
}3HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a7Z???r?i?Mozӛ???Unknown
4HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a7Z???r?iP?mQ????Unknown
i5HostMean"mean_squared_error/Mean(1       @9       @A       @I       @a7Z???r?i???`?????Unknown
?6HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a7Z???r?i,k?SM???Unknown
?7HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a7Z???r?i??F?1???Unknown
?8HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a7Z???r?i??9IW???Unknown
u9HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a7Z???b?i?.?3j???Unknown
`:HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a7Z???b?iv?-?|???Unknown
w;HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a7Z???b?i???&?????Unknown
u<HostMul"$gradient_tape/mean_squared_error/Mul(1      ??9      ??A      ??I      ??a7Z???b?i?<8 E????Unknown
u=HostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??a7Z???b?i??????Unknown
>HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      ??9      ??A      ??I      ??a7Z???b?iR?T?????Unknown
|?HostDivNoNan"&mean_squared_error/weighted_loss/value(1      ??9      ??A      ??I      ??a7Z???b?i?K??????Unknown
?@HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a7Z???b?i??qA????Unknown
?AHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a7Z???b?i?????????Unknown
4BHostIdentity"Identity(i?????????Unknown?
JCHostMul"&gradient_tape/mean_squared_error/mul_1(i?????????Unknown2CPU