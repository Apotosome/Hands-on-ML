	???#0@???#0@!???#0@	?S??WO@?S??WO@!?S??WO@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???#0@?J?ó??AO???*?@Y?l??3???rEagerKernelExecute 0*	???Q??@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate0??\??!?п?D?H@)]?,σ???1)f?AjH@:Preprocessing2U
Iterator::Model::ParallelMapV2v??2SZ??!p#???@@)v??2SZ??1p#???@@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{??B???!Dmé??%@){??B???1Dmé??%@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?d ??Ƶ?!v?9?b,@)?W?}W??1?\?E4?
@:Preprocessing2F
Iterator::ModelzZ?????!?)???ZA@)B?Ѫ?t??1?? !???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip:??l??!
k?6?RP@)?b.?z?1?s'_?_??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????G6w?!"??0?@??)????G6w?1"??0?@??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?d9	?/??!Y9???I@)?u?!HW?1h? X??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?S??WO@Ia?A?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?J?ó???J?ó??!?J?ó??      ??!       "      ??!       *      ??!       2	O???*?@O???*?@!O???*?@:      ??!       B      ??!       J	?l??3????l??3???!?l??3???R      ??!       Z	?l??3????l??3???!?l??3???b      ??!       JCPU_ONLYY?S??WO@b qa?A?X@