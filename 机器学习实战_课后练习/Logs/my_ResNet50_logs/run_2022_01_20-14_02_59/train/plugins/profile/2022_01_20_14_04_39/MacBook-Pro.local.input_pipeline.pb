	?????@?????@!?????@	??QD?k????QD?k??!??QD?k??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?????@H?Ȱ?7??A??h ??@Yffffff??rEagerKernelExecute 0*     ?g@    ???@2?
hIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2P7?A`??,@!߲???C@)7?A`??,@1߲???C@:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch7?|?5^?!@!??}#X8@)?|?5^?!@1??}#X8@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV23? ?rh?@!????>,@)? ?rh?@1????>,@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat@?ʡE?s)@!?Xؗ?yA@)L7?A`?@11R?c?6%@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4;?O??n??!?;"Ap?@);?O??n??1?;"Ap?@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMapD??x?&1??!x?A?@)\???(\??1???-.?@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinalityPF????x??!(??ݛ@))\???(??1?1??U??:Preprocessing2?
uIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCache6      @!?=3?5.@)X9??v???1ui??M??:Preprocessing2F
Iterator::Modelsh??|???!?T??-??)sh??|???1?T??-??:Preprocessing2?
yIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl#P??n@!???Z??,@)?x?&1??1????6???:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ShuffleAndRepeat::Prefetch::ParallelMapV2::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap[0]::TFRecord'???Mb??!???6????)???Mb??1???6????:Advanced file read2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch{?G?z??!??!???){?G?z??1??!???:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch??~j?t??!?ˁy????)??~j?t??1?ˁy????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??QD?k??I??KF??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H?Ȱ?7??H?Ȱ?7??!H?Ȱ?7??      ??!       "      ??!       *      ??!       2	??h ??@??h ??@!??h ??@:      ??!       B      ??!       J	ffffff??ffffff??!ffffff??R      ??!       Z	ffffff??ffffff??!ffffff??b      ??!       JCPU_ONLYY??QD?k??b q??KF??X@