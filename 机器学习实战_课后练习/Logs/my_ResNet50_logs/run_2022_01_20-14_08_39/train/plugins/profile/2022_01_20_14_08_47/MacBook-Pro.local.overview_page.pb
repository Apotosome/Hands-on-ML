?$	wi?a??@wi?a??@!wi?a??@	5/B?L???5/B?L???!5/B?L???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:wi?a??@3?뤾,??A0F$
u ?@Y??|?5^??rEagerKernelExecute 0*     ?@    ???@2?
hIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2P?|?5^1@!????	?B@)?|?5^1@1????	?B@:Preprocessing2?
uIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCache:ףp=
?$@!?1&Gj?6@)??/?d"@1??@??4@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV27?? ?r?@!2?4?.G0@)?? ?r?@12?4?.G0@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMapI;?O???@!??b:,@)??Q??@17?]???@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap[0]::TFRecord-9??v??@!?????@)9??v??@1?????@:Advanced file read2?
yIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl"NbX9?@!
?;&?E@)ffffff??1?C?x?
@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinalityP^?I+??!;ȭ|e@)?A`??"??1?Ѥ??@:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch;+????!?b	?O?@)+????1?b	?O?@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat@?????? @!??z??I@)/?$???1???V? @:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch\???(\??!??+Ƨm??)\???(\??1??+Ƨm??:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4T㥛? ??!?#`V???)T㥛? ??1?#`V???:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch????????!???o?ݻ?)????????1???o?ݻ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?p=
ף??!R??KY???){?G?z??1??YK??:Preprocessing2F
Iterator::Model?(\?????!?I+??+??){?G?zt?1??YK??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no95/B?L???I?[0???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	3?뤾,??3?뤾,??!3?뤾,??      ??!       "      ??!       *      ??!       2	0F$
u ?@0F$
u ?@!0F$
u ?@:      ??!       B      ??!       J	??|?5^????|?5^??!??|?5^??R      ??!       Z	??|?5^????|?5^??!??|?5^??b      ??!       JCPU_ONLYY5/B?L???b q?[0???X@Y      Y@qH??3s?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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