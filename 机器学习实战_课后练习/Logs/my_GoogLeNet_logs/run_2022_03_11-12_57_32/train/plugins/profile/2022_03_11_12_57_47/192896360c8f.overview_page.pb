? 	`??5?k@`??5?k@!`??5?k@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'`??5?k@??K8?F@1?j۰0j@I?u?+.?@r0*J+??@b??"/?A2?
PIterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2?ѐ?(??\@!???%?O@)ѐ?(??\@1???%?O@:Preprocessing2?
pIterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2?f?-?VB@!?cz.G4@)f?-?VB@1?cz.G4@:Preprocessing2y
AIterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch?o?KSL6@!0?,???(@)o?KSL6@10?,???(@:Preprocessing2?
?Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4?^????!T57???)^????1T57???:Preprocessing2o
7Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat??d?pu?7@!??"?*@)?ؘ????1??|?????:Preprocessing2?
?Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap[0]::TFRecord?=?K?ez??!m?}????)=?K?ez??1m?}????:Advanced file read2?
aIterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl?h?
?B@!?ĥa?4@)h>?nW??17)t???:Preprocessing2?
?Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality??+?,?@!IyS2???)??0~w??1>?'-mJ??:Preprocessing2?
]Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCache?MK???B@!H2?? 5@)??q?@??1`S?6s???:Preprocessing2?
?Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap?*??D??!y?n????)?W}w??1??@Bvk??:Preprocessing2E
Iterator::Root?+??E|??!?N?????)??2p@??1;??B"??:Preprocessing2\
%Iterator::Root::Prefetch::MapAndBatch?j????!"1x?ZN??)?j????1"1x?ZN??:Preprocessing2O
Iterator::Root::PrefetchiT?d???!??B?ͅ?)iT?d???1??B?ͅ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@???6j
@Q?2RK?,X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??K8?F@??K8?F@!??K8?F@      ??!       "	?j۰0j@?j۰0j@!?j۰0j@*      ??!       2      ??!       :	?u?+.?@?u?+.?@!?u?+.?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@???6j
@y?2RK?,X@?"d
8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter3?ѫ??!3?ѫ??0"b
6gradient_tape/model/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?o??ӣ?!RQ??7q??0"5
model/conv2d_2/Relu_FusedConv2Dߩa*???!?d?F!???"-
IteratorGetNext/_2_Recv&@?7??!?zzt??"b
7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInputL?еhD??!(4?????0"3
model/conv2d/Relu_FusedConv2D???aJ??!?q}???"V
5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad٦X?r-??!??\?+??"X
7gradient_tape/model/max_pooling2d_1/MaxPool/MaxPoolGradMaxPoolGrad:?.ˏ?!1?L9(??"6
model/conv2d_11/Relu_FusedConv2D???b??!=?l-W??"A
#gradient_tape/model/conv2d/ReluGradReluGrad?܀?*"??!#??>??Q      Y@Yz.9?P@a??Yz?@@q$?????y????[~X?"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 