	`??5?k@`??5?k@!`??5?k@      ??!       "h
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
@y?2RK?,X@