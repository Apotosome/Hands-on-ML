?	+????+????!+????	$I?$I?,@$I?$I?,@!$I?$I?,@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:+?????$??C??A???K7???Y?&1???rEagerKernelExecute 0*	     @c@2U
Iterator::Model::ParallelMapV2?? ?rh??!?O???F@)?? ?rh??1?O???F@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??v????!P????@@)9??v????1P????@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat{?G?z??!f?'?Y?)@)??~j?t??1?????(@:Preprocessing2F
Iterator::Model??~j?t??!?????H@)????Mb??1?S{?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!????8+@)?~j?t?x?1????8+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!?S{???)????MbP?1?S{???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 14.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t29.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9$I?$I?,@Iܶm۶mU@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?$??C???$??C??!?$??C??      ??!       "      ??!       *      ??!       2	???K7??????K7???!???K7???:      ??!       B      ??!       J	?&1????&1???!?&1???R      ??!       Z	?&1????&1???!?&1???b      ??!       JCPU_ONLYY$I?$I?,@b qܶm۶mU@Y      Y@q??k?? @"?	
both?Your program is MODERATELY input-bound because 14.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t29.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQ2"CPU: B 