?	Fx{"'@Fx{"'@!Fx{"'@	????^???????^???!????^???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:Fx{"'@?9??ȭ?A8?a?A?&@YW?Y????rEagerKernelExecute 0*	??Q??`@2U
Iterator::Model::ParallelMapV2p??;???!?$?׎C@)p??;???1?$?׎C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??\5ϩ?!c???+C@)-??罹?1???@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeats?9>Z???!ש?ג?)@)7???????1?:х?&@:Preprocessing2F
Iterator::Model????UG??!TƫܭRF@)?X?? ~?1.?̰@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??7?-:y?!?^?jL?@)??7?-:y?1?^?jL?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??;Ų?!?9T#R?K@)??[;Qr?1?o
N??
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???o
+e?!gxːb6??)???o
+e?1gxːb6??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????y??!;?x8?C@)??-YU?1ܵA$jz??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????^???I??>B??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?9??ȭ??9??ȭ?!?9??ȭ?      ??!       "      ??!       *      ??!       2	8?a?A?&@8?a?A?&@!8?a?A?&@:      ??!       B      ??!       J	W?Y????W?Y????!W?Y????R      ??!       Z	W?Y????W?Y????!W?Y????b      ??!       JCPU_ONLYY????^???b q??>B??X@Y      Y@q???J(^??"?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 