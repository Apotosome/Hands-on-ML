	=??ڵ'@=??ڵ'@!=??ڵ'@	????Db??????Db??!????Db??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:=??ڵ'@???????A?#?w~Y'@Y?Ŧ?B ??rEagerKernelExecute 0*	F????@b@2U
Iterator::Model::ParallelMapV2?{?q??!8w6JJ?D@)?{?q??18w6JJ?D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????'??!???4)B@)?k^?Y-??1WQ??/+@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty?ՏM???!??p?ƭ*@)7????1Zc??h%@:Preprocessing2F
Iterator::Model?K?uT??!?(??-G@)n?8)?{|?1?e&u@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-^,??w?!??y?H?@)-^,??w?1??y?H?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipeRC???!E??'?J@)????p?1Ҭ^.ې@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorV??Dׅo?!F?	?@)V??Dׅo?1F?	?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? ????!??????B@)?#EdX?[?1	ӷ?p???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????Db??I??bv;?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????????!???????      ??!       "      ??!       *      ??!       2	?#?w~Y'@?#?w~Y'@!?#?w~Y'@:      ??!       B      ??!       J	?Ŧ?B ???Ŧ?B ??!?Ŧ?B ??R      ??!       Z	?Ŧ?B ???Ŧ?B ??!?Ŧ?B ??b      ??!       JCPU_ONLYY????Db??b q??bv;?X@