	.??S?@.??S?@!.??S?@	?1??aS@?1??aS@!?1??aS@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:.??S?@;oc?#ջ?A?<??z@Y?
?F??rEagerKernelExecute 0*	/?$??i@2U
Iterator::Model::ParallelMapV2??}?p??!????J@)??}?p??1????J@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??d??!?J?T8@)??H¾??1??@?]?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM!u;???!???VR?)@)N?E? ??1?u?X9?$@:Preprocessing2F
Iterator::Model?'v?U??!G?????M@)?ȓ?k&??1?????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicejkD0.}?!??-?˵@)jkD0.}?1??-?˵@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipuʣaQ??!?74>D@)y?ߢ??v?1?o????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??i? ?s?!A?m?c?@)??i? ?s?1A?m?c?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapU[rP??!???#?8@)??H?}]?1^>Fy??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?1??aS@Ip?:?d%X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;oc?#ջ?;oc?#ջ?!;oc?#ջ?      ??!       "      ??!       *      ??!       2	?<??z@?<??z@!?<??z@:      ??!       B      ??!       J	?
?F???
?F??!?
?F??R      ??!       Z	?
?F???
?F??!?
?F??b      ??!       JCPU_ONLYY?1??aS@b qp?:?d%X@