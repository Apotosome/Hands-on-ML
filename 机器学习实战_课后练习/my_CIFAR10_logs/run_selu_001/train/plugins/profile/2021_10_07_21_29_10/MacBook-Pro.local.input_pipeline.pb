	Z?b+hJ@Z?b+hJ@!Z?b+hJ@	U?@U?@!U?@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:Z?b+hJ@?^Pj??A֏M?#>@Y3???yS??rEagerKernelExecute 0*	?K7?A|k@2U
Iterator::Model::ParallelMapV2?EИ??!
?,&/C@)?EИ??1
?,&/C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?P??9??!???Z?<>@)hA(??h??1Ȋ]??;@:Preprocessing2F
Iterator::Model?)??z???!???L@)?1ZGU??1????1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeataQ??l??! ?5??#@)w|??ّ?1??)2??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??A?}?!?a,??	@)??A?}?1?a,??	@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipHG?ŧ??!08??x?E@)_?Q?{?1?[Ԍ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorR臭??l?!??Wc??)R臭??l?1??Wc??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy???????!!&?We1?@)??2?68a?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9U?@Im:?XZX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?^Pj???^Pj??!?^Pj??      ??!       "      ??!       *      ??!       2	֏M?#>@֏M?#>@!֏M?#>@:      ??!       B      ??!       J	3???yS??3???yS??!3???yS??R      ??!       Z	3???yS??3???yS??!3???yS??b      ??!       JCPU_ONLYYU?@b qm:?XZX@