	.??e?_@.??e?_@!.??e?_@	H}???{??H}???{??!H}???{??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:.??e?_@??G?ֺ?A<jL???@Y???H????rEagerKernelExecute 0*	s?Vn_@2U
Iterator::Model::ParallelMapV2O??e?c??!Q?|??6@)O??e?c??1Q?|??6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?J?4??!?z?4??:@)JC?B?Y??1?y??,?6@:Preprocessing2F
Iterator::Model??????!=?mE@)?g\W̘?1*??C3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?V`??V??!??:2?W8@)?yq????1?fNj?f0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????}r??!???@)????}r??1???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?|??z???!?*????L@)??eO{?1:????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?SH?9t?!0	?O?k@)?SH?9t?10	?O?k@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap@Qٰ????!5???9@)?_??s`?1!6?o????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9H}???{??I*?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??G?ֺ???G?ֺ?!??G?ֺ?      ??!       "      ??!       *      ??!       2	<jL???@<jL???@!<jL???@:      ??!       B      ??!       J	???H???????H????!???H????R      ??!       Z	???H???????H????!???H????b      ??!       JCPU_ONLYYH}???{??b q*?X@