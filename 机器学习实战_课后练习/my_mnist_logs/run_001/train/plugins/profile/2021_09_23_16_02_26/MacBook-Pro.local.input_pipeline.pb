	`??"????`??"????!`??"????	???O?@???O?@!???O?@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:`??"????????Mb??A?I+???YbX9?ȶ?rEagerKernelExecute 0*	     ?a@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Q?????!??O$??H@)?Q?????1??O$??H@:Preprocessing2U
Iterator::Model::ParallelMapV2????????!?X???1@)????????1?X???1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty?&1???!????3@)Zd;?O???1?pJ??O0@:Preprocessing2F
Iterator::Model?l??????!Y???=:@)?~j?t???1??
br!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!??
br@)?~j?t?x?1??
br@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!?'Ni^@){?G?zt?1?'Ni^@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s4.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???O?@Its@?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????Mb??????Mb??!????Mb??      ??!       "      ??!       *      ??!       2	?I+????I+???!?I+???:      ??!       B      ??!       J	bX9?ȶ?bX9?ȶ?!bX9?ȶ?R      ??!       Z	bX9?ȶ?bX9?ȶ?!bX9?ȶ?b      ??!       JCPU_ONLYY???O?@b qts@?W@