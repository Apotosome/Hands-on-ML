? 	?????w@?????w@!?????w@      ??!       "h
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
????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@???????yå??h?X@?"l
@gradient_tape/model/feature/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!??????0"l
@gradient_tape/model/feature/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter {???V??!????????0"=
model/feature/conv2d_1/Relu_FusedConv2D?]̽???!P?^? ??"j
?gradient_tape/model/feature/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput??3R??!vֱ?,???0"l
@gradient_tape/model/feature/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?l?c?5??!̘WbL???0"=
model/feature/conv2d_3/Relu_FusedConv2D̹?+>???!?'???"j
?gradient_tape/model/feature/conv2d_3/Conv2D/Conv2DBackpropInputConv2DBackpropInput֔M?????!??;????0"l
@gradient_tape/model/feature/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter ?-?fR??!?z%?i??0"l
@gradient_tape/model/feature/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter# ??(??!?|u?\,??0"=
model/feature/conv2d_6/Relu_FusedConv2D???\????!]?B?V???Q      Y@Y?	??0!S@a???={7@q\YDs???y!?3:?	L?"?	
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