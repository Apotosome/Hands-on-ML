?$	?A	3]L?@?A	3]L?@!?A	3]L?@	]j????|?]j????|?!]j????|?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?A	3]L?@[|
????A???YK?@Y????Mb??rEagerKernelExecute 0*      `@    ???@2?
hIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2@%??C@!?????@@)%??C@1?????@@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2!?rh??|@!?"?5m?=@)?rh??|@1?"?5m?=@:Preprocessing2?
uIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCache&`??"?y@!????D?>@)?O??n @17???"@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4ˡE?????!??EG? @)ˡE?????1??EG? @:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap:?V-??!?=?9?] @)????Mb??11@TJ??@:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch7d;?O????!2?a1?F@)d;?O????12?a1?F@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat@?p=
ף??!W+??8?@)j?t???1J?#?@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality@???K7?@!?	?h_$@)R???Q??1????A??:Preprocessing2?
yIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl +???@!?????5@)??(\????19:??G??:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap[0]::TFRecord????Mb??!4?ݞ???)????Mb??14?ݞ???:Advanced file read2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch??MbX??!<??m?q??)??MbX??1<??m?q??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch#??~j???!?aq ??)#??~j???1?aq ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???Mb??!?aU?????)9??v????1??'BI???:Preprocessing2F
Iterator::Model??MbX??!<??m?q??){?G?zt?1???F?ʗ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9]j????|?I.?}4??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	[|
????[|
????![|
????      ??!       "      ??!       *      ??!       2	???YK?@???YK?@!???YK?@:      ??!       B      ??!       J	????Mb??????Mb??!????Mb??R      ??!       Z	????Mb??????Mb??!????Mb??b      ??!       JCPU_ONLYY]j????|?b q.?}4??X@Y      Y@qo9??*}??"?
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