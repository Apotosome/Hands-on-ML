	$?]J]?@$?]J]?@!$?]J]?@	?p?y?q@?p?y?q@!?p?y?q@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:$?]J]?@J?\??A????@YX?Q???rEagerKernelExecute 0*	+???~@2U
Iterator::Model::ParallelMapV2?<G仔??!{9??K@)?<G仔??1{9??K@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg|_\????!j=?`B@)w???????1	???*?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??tޙ?!,?]?[z@)??tޙ?1,?]?[z@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????b??!?+\j?@)?=$|?o??1	?Ȁx
@:Preprocessing2F
Iterator::Model?Z?}??!rͼ??FM@)|??8G??1]?<?m@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?k^?Y-??!?2CT)?D@)Y??L/1v?1ci?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorQ?[??g?!&D?mǫ??)Q?[??g?1&D?mǫ??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??{??!Ry?Y?AB@)??IӠh^?1??p???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?p?y?q@I?(cX??W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	J?\??J?\??!J?\??      ??!       "      ??!       *      ??!       2	????@????@!????@:      ??!       B      ??!       J	X?Q???X?Q???!X?Q???R      ??!       Z	X?Q???X?Q???!X?Q???b      ??!       JCPU_ONLYY?p?y?q@b q?(cX??W@