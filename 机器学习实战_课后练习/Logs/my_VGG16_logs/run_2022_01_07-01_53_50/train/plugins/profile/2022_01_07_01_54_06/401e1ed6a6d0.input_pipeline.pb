	?????w@?????w@!?????w@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?????w@w? ݗs	@1rP??w@I8N
????r0*?G?z@?@??"????@2?
PIterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2@E?J=I@!??????S@)E?J=I@1??????S@:Preprocessing2?
pIterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2@@???v#@!???
?.@)@???v#@1???
?.@:Preprocessing2n
7Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat@??D?a??!???K
???);?%8????1?K쥍??:Preprocessing2?
?Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4@y?@e????!???sZc??)y?@e????1???sZc??:Preprocessing2?
?Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap[0]::TFRecord@????y???!V$??????)????y???1V$??????:Advanced file read2?
?Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality@'?WJ??!ܢn!???)[닄????1U?? ??:Preprocessing2?
aIterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl@?#G:?#@!????d/@)X?%?????1"{NG??:Preprocessing2x
AIterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch@?б???!?g~?9??)?б???1?g~?9??:Preprocessing2?
]Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCache@_\??/$@!]?!5?/@)-????1?g? ???:Preprocessing2?
?Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap@s??????!~ݔPn??)k?ѯ???1 _?@????:Preprocessing2O
Iterator::Root::Prefetcho*Ral!??!M?s?Bv??)o*Ral!??1M?s?Bv??:Preprocessing2\
%Iterator::Root::Prefetch::MapAndBatchގpZ????!?	????)ގpZ????1?	????:Preprocessing2E
Iterator::Root?+??ص??!?????r??)?^f?(??1/Q??o??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@???????Qå??h?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	w? ݗs	@w? ݗs	@!w? ݗs	@      ??!       "	rP??w@rP??w@!rP??w@*      ??!       2      ??!       :	8N
????8N
????!8N
????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@???????yå??h?X@