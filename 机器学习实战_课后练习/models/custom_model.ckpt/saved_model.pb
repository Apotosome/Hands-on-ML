Ä	
ï¿
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
;
Elu
features"T
activations"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ýÄ

!residual_regressor/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!residual_regressor/dense_5/kernel

5residual_regressor/dense_5/kernel/Read/ReadVariableOpReadVariableOp!residual_regressor/dense_5/kernel*
_output_shapes

:*
dtype0

residual_regressor/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!residual_regressor/dense_5/bias

3residual_regressor/dense_5/bias/Read/ReadVariableOpReadVariableOpresidual_regressor/dense_5/bias*
_output_shapes
:*
dtype0
 
"residual_regressor/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"residual_regressor/dense_10/kernel

6residual_regressor/dense_10/kernel/Read/ReadVariableOpReadVariableOp"residual_regressor/dense_10/kernel*
_output_shapes

:*
dtype0

 residual_regressor/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" residual_regressor/dense_10/bias

4residual_regressor/dense_10/bias/Read/ReadVariableOpReadVariableOp residual_regressor/dense_10/bias*
_output_shapes
:*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
¼
0residual_regressor/residual_block/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20residual_regressor/residual_block/dense_6/kernel
µ
Dresidual_regressor/residual_block/dense_6/kernel/Read/ReadVariableOpReadVariableOp0residual_regressor/residual_block/dense_6/kernel*
_output_shapes

:*
dtype0
´
.residual_regressor/residual_block/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.residual_regressor/residual_block/dense_6/bias
­
Bresidual_regressor/residual_block/dense_6/bias/Read/ReadVariableOpReadVariableOp.residual_regressor/residual_block/dense_6/bias*
_output_shapes
:*
dtype0
¼
0residual_regressor/residual_block/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*A
shared_name20residual_regressor/residual_block/dense_7/kernel
µ
Dresidual_regressor/residual_block/dense_7/kernel/Read/ReadVariableOpReadVariableOp0residual_regressor/residual_block/dense_7/kernel*
_output_shapes

:*
dtype0
´
.residual_regressor/residual_block/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.residual_regressor/residual_block/dense_7/bias
­
Bresidual_regressor/residual_block/dense_7/bias/Read/ReadVariableOpReadVariableOp.residual_regressor/residual_block/dense_7/bias*
_output_shapes
:*
dtype0
À
2residual_regressor/residual_block_1/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42residual_regressor/residual_block_1/dense_8/kernel
¹
Fresidual_regressor/residual_block_1/dense_8/kernel/Read/ReadVariableOpReadVariableOp2residual_regressor/residual_block_1/dense_8/kernel*
_output_shapes

:*
dtype0
¸
0residual_regressor/residual_block_1/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20residual_regressor/residual_block_1/dense_8/bias
±
Dresidual_regressor/residual_block_1/dense_8/bias/Read/ReadVariableOpReadVariableOp0residual_regressor/residual_block_1/dense_8/bias*
_output_shapes
:*
dtype0
À
2residual_regressor/residual_block_1/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42residual_regressor/residual_block_1/dense_9/kernel
¹
Fresidual_regressor/residual_block_1/dense_9/kernel/Read/ReadVariableOpReadVariableOp2residual_regressor/residual_block_1/dense_9/kernel*
_output_shapes

:*
dtype0
¸
0residual_regressor/residual_block_1/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20residual_regressor/residual_block_1/dense_9/bias
±
Dresidual_regressor/residual_block_1/dense_9/bias/Read/ReadVariableOpReadVariableOp0residual_regressor/residual_block_1/dense_9/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
®
)Nadam/residual_regressor/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Nadam/residual_regressor/dense_5/kernel/m
§
=Nadam/residual_regressor/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp)Nadam/residual_regressor/dense_5/kernel/m*
_output_shapes

:*
dtype0
¦
'Nadam/residual_regressor/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Nadam/residual_regressor/dense_5/bias/m

;Nadam/residual_regressor/dense_5/bias/m/Read/ReadVariableOpReadVariableOp'Nadam/residual_regressor/dense_5/bias/m*
_output_shapes
:*
dtype0
°
*Nadam/residual_regressor/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Nadam/residual_regressor/dense_10/kernel/m
©
>Nadam/residual_regressor/dense_10/kernel/m/Read/ReadVariableOpReadVariableOp*Nadam/residual_regressor/dense_10/kernel/m*
_output_shapes

:*
dtype0
¨
(Nadam/residual_regressor/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Nadam/residual_regressor/dense_10/bias/m
¡
<Nadam/residual_regressor/dense_10/bias/m/Read/ReadVariableOpReadVariableOp(Nadam/residual_regressor/dense_10/bias/m*
_output_shapes
:*
dtype0
Ì
8Nadam/residual_regressor/residual_block/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Nadam/residual_regressor/residual_block/dense_6/kernel/m
Å
LNadam/residual_regressor/residual_block/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp8Nadam/residual_regressor/residual_block/dense_6/kernel/m*
_output_shapes

:*
dtype0
Ä
6Nadam/residual_regressor/residual_block/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Nadam/residual_regressor/residual_block/dense_6/bias/m
½
JNadam/residual_regressor/residual_block/dense_6/bias/m/Read/ReadVariableOpReadVariableOp6Nadam/residual_regressor/residual_block/dense_6/bias/m*
_output_shapes
:*
dtype0
Ì
8Nadam/residual_regressor/residual_block/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Nadam/residual_regressor/residual_block/dense_7/kernel/m
Å
LNadam/residual_regressor/residual_block/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp8Nadam/residual_regressor/residual_block/dense_7/kernel/m*
_output_shapes

:*
dtype0
Ä
6Nadam/residual_regressor/residual_block/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Nadam/residual_regressor/residual_block/dense_7/bias/m
½
JNadam/residual_regressor/residual_block/dense_7/bias/m/Read/ReadVariableOpReadVariableOp6Nadam/residual_regressor/residual_block/dense_7/bias/m*
_output_shapes
:*
dtype0
Ð
:Nadam/residual_regressor/residual_block_1/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:Nadam/residual_regressor/residual_block_1/dense_8/kernel/m
É
NNadam/residual_regressor/residual_block_1/dense_8/kernel/m/Read/ReadVariableOpReadVariableOp:Nadam/residual_regressor/residual_block_1/dense_8/kernel/m*
_output_shapes

:*
dtype0
È
8Nadam/residual_regressor/residual_block_1/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Nadam/residual_regressor/residual_block_1/dense_8/bias/m
Á
LNadam/residual_regressor/residual_block_1/dense_8/bias/m/Read/ReadVariableOpReadVariableOp8Nadam/residual_regressor/residual_block_1/dense_8/bias/m*
_output_shapes
:*
dtype0
Ð
:Nadam/residual_regressor/residual_block_1/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:Nadam/residual_regressor/residual_block_1/dense_9/kernel/m
É
NNadam/residual_regressor/residual_block_1/dense_9/kernel/m/Read/ReadVariableOpReadVariableOp:Nadam/residual_regressor/residual_block_1/dense_9/kernel/m*
_output_shapes

:*
dtype0
È
8Nadam/residual_regressor/residual_block_1/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Nadam/residual_regressor/residual_block_1/dense_9/bias/m
Á
LNadam/residual_regressor/residual_block_1/dense_9/bias/m/Read/ReadVariableOpReadVariableOp8Nadam/residual_regressor/residual_block_1/dense_9/bias/m*
_output_shapes
:*
dtype0
®
)Nadam/residual_regressor/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Nadam/residual_regressor/dense_5/kernel/v
§
=Nadam/residual_regressor/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp)Nadam/residual_regressor/dense_5/kernel/v*
_output_shapes

:*
dtype0
¦
'Nadam/residual_regressor/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Nadam/residual_regressor/dense_5/bias/v

;Nadam/residual_regressor/dense_5/bias/v/Read/ReadVariableOpReadVariableOp'Nadam/residual_regressor/dense_5/bias/v*
_output_shapes
:*
dtype0
°
*Nadam/residual_regressor/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Nadam/residual_regressor/dense_10/kernel/v
©
>Nadam/residual_regressor/dense_10/kernel/v/Read/ReadVariableOpReadVariableOp*Nadam/residual_regressor/dense_10/kernel/v*
_output_shapes

:*
dtype0
¨
(Nadam/residual_regressor/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Nadam/residual_regressor/dense_10/bias/v
¡
<Nadam/residual_regressor/dense_10/bias/v/Read/ReadVariableOpReadVariableOp(Nadam/residual_regressor/dense_10/bias/v*
_output_shapes
:*
dtype0
Ì
8Nadam/residual_regressor/residual_block/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Nadam/residual_regressor/residual_block/dense_6/kernel/v
Å
LNadam/residual_regressor/residual_block/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp8Nadam/residual_regressor/residual_block/dense_6/kernel/v*
_output_shapes

:*
dtype0
Ä
6Nadam/residual_regressor/residual_block/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Nadam/residual_regressor/residual_block/dense_6/bias/v
½
JNadam/residual_regressor/residual_block/dense_6/bias/v/Read/ReadVariableOpReadVariableOp6Nadam/residual_regressor/residual_block/dense_6/bias/v*
_output_shapes
:*
dtype0
Ì
8Nadam/residual_regressor/residual_block/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*I
shared_name:8Nadam/residual_regressor/residual_block/dense_7/kernel/v
Å
LNadam/residual_regressor/residual_block/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp8Nadam/residual_regressor/residual_block/dense_7/kernel/v*
_output_shapes

:*
dtype0
Ä
6Nadam/residual_regressor/residual_block/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Nadam/residual_regressor/residual_block/dense_7/bias/v
½
JNadam/residual_regressor/residual_block/dense_7/bias/v/Read/ReadVariableOpReadVariableOp6Nadam/residual_regressor/residual_block/dense_7/bias/v*
_output_shapes
:*
dtype0
Ð
:Nadam/residual_regressor/residual_block_1/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:Nadam/residual_regressor/residual_block_1/dense_8/kernel/v
É
NNadam/residual_regressor/residual_block_1/dense_8/kernel/v/Read/ReadVariableOpReadVariableOp:Nadam/residual_regressor/residual_block_1/dense_8/kernel/v*
_output_shapes

:*
dtype0
È
8Nadam/residual_regressor/residual_block_1/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Nadam/residual_regressor/residual_block_1/dense_8/bias/v
Á
LNadam/residual_regressor/residual_block_1/dense_8/bias/v/Read/ReadVariableOpReadVariableOp8Nadam/residual_regressor/residual_block_1/dense_8/bias/v*
_output_shapes
:*
dtype0
Ð
:Nadam/residual_regressor/residual_block_1/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:Nadam/residual_regressor/residual_block_1/dense_9/kernel/v
É
NNadam/residual_regressor/residual_block_1/dense_9/kernel/v/Read/ReadVariableOpReadVariableOp:Nadam/residual_regressor/residual_block_1/dense_9/kernel/v*
_output_shapes

:*
dtype0
È
8Nadam/residual_regressor/residual_block_1/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Nadam/residual_regressor/residual_block_1/dense_9/bias/v
Á
LNadam/residual_regressor/residual_block_1/dense_9/bias/v/Read/ReadVariableOpReadVariableOp8Nadam/residual_regressor/residual_block_1/dense_9/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ÀI
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ûH
valueñHBîH BçH
¸

hidden
ResidualBlock_1
ResidualBlock_2
outputLayer
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
^

hidden
trainable_variables
	variables
regularization_losses
	keras_api
^

hidden
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
¹
!iter

"beta_1

#beta_2
	$decay
%learning_rate
&momentum_cachemumvmwmx'my(mz)m{*m|+m},m~-m.mvvvv'v(v)v*v+v,v-v.v
V
0
1
'2
(3
)4
*5
+6
,7
-8
.9
10
11
V
0
1
'2
(3
)4
*5
+6
,7
-8
.9
10
11
 
­
trainable_variables
	variables
/non_trainable_variables

0layers
1layer_metrics
2metrics
regularization_losses
3layer_regularization_losses
 
_]
VARIABLE_VALUE!residual_regressor/dense_5/kernel(hidden/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEresidual_regressor/dense_5/bias&hidden/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
	variables
4non_trainable_variables

5layers
6layer_metrics
7metrics
regularization_losses
8layer_regularization_losses

90
:1

'0
(1
)2
*3

'0
(1
)2
*3
 
­
trainable_variables
	variables
;non_trainable_variables

<layers
=layer_metrics
>metrics
regularization_losses
?layer_regularization_losses

@0
A1

+0
,1
-2
.3

+0
,1
-2
.3
 
­
trainable_variables
	variables
Bnon_trainable_variables

Clayers
Dlayer_metrics
Emetrics
regularization_losses
Flayer_regularization_losses
ec
VARIABLE_VALUE"residual_regressor/dense_10/kernel-outputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE residual_regressor/dense_10/bias+outputLayer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
	variables
Gnon_trainable_variables

Hlayers
Ilayer_metrics
Jmetrics
regularization_losses
Klayer_regularization_losses
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0residual_regressor/residual_block/dense_6/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE.residual_regressor/residual_block/dense_6/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0residual_regressor/residual_block/dense_7/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE.residual_regressor/residual_block/dense_7/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2residual_regressor/residual_block_1/dense_8/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0residual_regressor/residual_block_1/dense_8/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2residual_regressor/residual_block_1/dense_9/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0residual_regressor/residual_block_1/dense_9/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3
 

L0
 
 
 
 
 
 
h

'kernel
(bias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
h

)kernel
*bias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
 

90
:1
 
 
 
h

+kernel
,bias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
h

-kernel
.bias
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
 

@0
A1
 
 
 
 
 
 
 
 
4
	]total
	^count
_	variables
`	keras_api

'0
(1

'0
(1
 
­
Mtrainable_variables
N	variables
anon_trainable_variables

blayers
clayer_metrics
dmetrics
Oregularization_losses
elayer_regularization_losses

)0
*1

)0
*1
 
­
Qtrainable_variables
R	variables
fnon_trainable_variables

glayers
hlayer_metrics
imetrics
Sregularization_losses
jlayer_regularization_losses

+0
,1

+0
,1
 
­
Utrainable_variables
V	variables
knon_trainable_variables

llayers
mlayer_metrics
nmetrics
Wregularization_losses
olayer_regularization_losses

-0
.1

-0
.1
 
­
Ytrainable_variables
Z	variables
pnon_trainable_variables

qlayers
rlayer_metrics
smetrics
[regularization_losses
tlayer_regularization_losses
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

]0
^1

_	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

VARIABLE_VALUE)Nadam/residual_regressor/dense_5/kernel/mDhidden/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Nadam/residual_regressor/dense_5/bias/mBhidden/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Nadam/residual_regressor/dense_10/kernel/mIoutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Nadam/residual_regressor/dense_10/bias/mGoutputLayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Nadam/residual_regressor/residual_block/dense_6/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Nadam/residual_regressor/residual_block/dense_6/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Nadam/residual_regressor/residual_block/dense_7/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Nadam/residual_regressor/residual_block/dense_7/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Nadam/residual_regressor/residual_block_1/dense_8/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Nadam/residual_regressor/residual_block_1/dense_8/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Nadam/residual_regressor/residual_block_1/dense_9/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Nadam/residual_regressor/residual_block_1/dense_9/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)Nadam/residual_regressor/dense_5/kernel/vDhidden/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Nadam/residual_regressor/dense_5/bias/vBhidden/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Nadam/residual_regressor/dense_10/kernel/vIoutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(Nadam/residual_regressor/dense_10/bias/vGoutputLayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Nadam/residual_regressor/residual_block/dense_6/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Nadam/residual_regressor/residual_block/dense_6/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Nadam/residual_regressor/residual_block/dense_7/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE6Nadam/residual_regressor/residual_block/dense_7/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Nadam/residual_regressor/residual_block_1/dense_8/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Nadam/residual_regressor/residual_block_1/dense_8/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE:Nadam/residual_regressor/residual_block_1/dense_9/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE8Nadam/residual_regressor/residual_block_1/dense_9/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ã
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!residual_regressor/dense_5/kernelresidual_regressor/dense_5/bias0residual_regressor/residual_block/dense_6/kernel.residual_regressor/residual_block/dense_6/bias0residual_regressor/residual_block/dense_7/kernel.residual_regressor/residual_block/dense_7/bias2residual_regressor/residual_block_1/dense_8/kernel0residual_regressor/residual_block_1/dense_8/bias2residual_regressor/residual_block_1/dense_9/kernel0residual_regressor/residual_block_1/dense_9/bias"residual_regressor/dense_10/kernel residual_regressor/dense_10/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_97475
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
²
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5residual_regressor/dense_5/kernel/Read/ReadVariableOp3residual_regressor/dense_5/bias/Read/ReadVariableOp6residual_regressor/dense_10/kernel/Read/ReadVariableOp4residual_regressor/dense_10/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOpDresidual_regressor/residual_block/dense_6/kernel/Read/ReadVariableOpBresidual_regressor/residual_block/dense_6/bias/Read/ReadVariableOpDresidual_regressor/residual_block/dense_7/kernel/Read/ReadVariableOpBresidual_regressor/residual_block/dense_7/bias/Read/ReadVariableOpFresidual_regressor/residual_block_1/dense_8/kernel/Read/ReadVariableOpDresidual_regressor/residual_block_1/dense_8/bias/Read/ReadVariableOpFresidual_regressor/residual_block_1/dense_9/kernel/Read/ReadVariableOpDresidual_regressor/residual_block_1/dense_9/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp=Nadam/residual_regressor/dense_5/kernel/m/Read/ReadVariableOp;Nadam/residual_regressor/dense_5/bias/m/Read/ReadVariableOp>Nadam/residual_regressor/dense_10/kernel/m/Read/ReadVariableOp<Nadam/residual_regressor/dense_10/bias/m/Read/ReadVariableOpLNadam/residual_regressor/residual_block/dense_6/kernel/m/Read/ReadVariableOpJNadam/residual_regressor/residual_block/dense_6/bias/m/Read/ReadVariableOpLNadam/residual_regressor/residual_block/dense_7/kernel/m/Read/ReadVariableOpJNadam/residual_regressor/residual_block/dense_7/bias/m/Read/ReadVariableOpNNadam/residual_regressor/residual_block_1/dense_8/kernel/m/Read/ReadVariableOpLNadam/residual_regressor/residual_block_1/dense_8/bias/m/Read/ReadVariableOpNNadam/residual_regressor/residual_block_1/dense_9/kernel/m/Read/ReadVariableOpLNadam/residual_regressor/residual_block_1/dense_9/bias/m/Read/ReadVariableOp=Nadam/residual_regressor/dense_5/kernel/v/Read/ReadVariableOp;Nadam/residual_regressor/dense_5/bias/v/Read/ReadVariableOp>Nadam/residual_regressor/dense_10/kernel/v/Read/ReadVariableOp<Nadam/residual_regressor/dense_10/bias/v/Read/ReadVariableOpLNadam/residual_regressor/residual_block/dense_6/kernel/v/Read/ReadVariableOpJNadam/residual_regressor/residual_block/dense_6/bias/v/Read/ReadVariableOpLNadam/residual_regressor/residual_block/dense_7/kernel/v/Read/ReadVariableOpJNadam/residual_regressor/residual_block/dense_7/bias/v/Read/ReadVariableOpNNadam/residual_regressor/residual_block_1/dense_8/kernel/v/Read/ReadVariableOpLNadam/residual_regressor/residual_block_1/dense_8/bias/v/Read/ReadVariableOpNNadam/residual_regressor/residual_block_1/dense_9/kernel/v/Read/ReadVariableOpLNadam/residual_regressor/residual_block_1/dense_9/bias/v/Read/ReadVariableOpConst*9
Tin2
02.	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_97733
½
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!residual_regressor/dense_5/kernelresidual_regressor/dense_5/bias"residual_regressor/dense_10/kernel residual_regressor/dense_10/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cache0residual_regressor/residual_block/dense_6/kernel.residual_regressor/residual_block/dense_6/bias0residual_regressor/residual_block/dense_7/kernel.residual_regressor/residual_block/dense_7/bias2residual_regressor/residual_block_1/dense_8/kernel0residual_regressor/residual_block_1/dense_8/bias2residual_regressor/residual_block_1/dense_9/kernel0residual_regressor/residual_block_1/dense_9/biastotalcount)Nadam/residual_regressor/dense_5/kernel/m'Nadam/residual_regressor/dense_5/bias/m*Nadam/residual_regressor/dense_10/kernel/m(Nadam/residual_regressor/dense_10/bias/m8Nadam/residual_regressor/residual_block/dense_6/kernel/m6Nadam/residual_regressor/residual_block/dense_6/bias/m8Nadam/residual_regressor/residual_block/dense_7/kernel/m6Nadam/residual_regressor/residual_block/dense_7/bias/m:Nadam/residual_regressor/residual_block_1/dense_8/kernel/m8Nadam/residual_regressor/residual_block_1/dense_8/bias/m:Nadam/residual_regressor/residual_block_1/dense_9/kernel/m8Nadam/residual_regressor/residual_block_1/dense_9/bias/m)Nadam/residual_regressor/dense_5/kernel/v'Nadam/residual_regressor/dense_5/bias/v*Nadam/residual_regressor/dense_10/kernel/v(Nadam/residual_regressor/dense_10/bias/v8Nadam/residual_regressor/residual_block/dense_6/kernel/v6Nadam/residual_regressor/residual_block/dense_6/bias/v8Nadam/residual_regressor/residual_block/dense_7/kernel/v6Nadam/residual_regressor/residual_block/dense_7/bias/v:Nadam/residual_regressor/residual_block_1/dense_8/kernel/v8Nadam/residual_regressor/residual_block_1/dense_8/bias/v:Nadam/residual_regressor/residual_block_1/dense_9/kernel/v8Nadam/residual_regressor/residual_block_1/dense_9/bias/v*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_97875Ó
Ú
Ì
I__inference_residual_block_layer_call_and_return_conditional_losses_97289

inputs8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:
identity¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¥
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¡
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAddm
dense_6/EluEludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Elu¥
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMuldense_6/Elu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMul¤
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp¡
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAddm
dense_7/EluEludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/Eluh
addAddV2dense_7/Elu:activations:0inputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addÝ
IdentityIdentityadd:z:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


(__inference_dense_10_layer_call_fn_97568

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_973532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


'__inference_dense_5_layer_call_fn_97484

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_972642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï	
ô
C__inference_dense_10_layer_call_and_return_conditional_losses_97578

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
Î
K__inference_residual_block_1_layer_call_and_return_conditional_losses_97333

inputs8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¥
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/BiasAddm
dense_8/EluEludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/Elu¥
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMuldense_8/Elu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp¡
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/BiasAddm
dense_9/EluEludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/Eluh
addAddV2dense_9/Elu:activations:0inputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addÝ
IdentityIdentityadd:z:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö

¯
2__inference_residual_regressor_layer_call_fn_97390
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_residual_regressor_layer_call_and_return_conditional_losses_973602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ú
Ì
I__inference_residual_block_layer_call_and_return_conditional_losses_97527

inputs8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:
identity¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¥
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¡
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAddm
dense_6/EluEludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Elu¥
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMuldense_6/Elu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMul¤
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp¡
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAddm
dense_7/EluEludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/Eluh
addAddV2dense_7/Elu:activations:0inputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addÝ
IdentityIdentityadd:z:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÕÀ
î
 __inference__wrapped_model_97249
input_1K
9residual_regressor_dense_5_matmul_readvariableop_resource:H
:residual_regressor_dense_5_biasadd_readvariableop_resource:Z
Hresidual_regressor_residual_block_dense_6_matmul_readvariableop_resource:W
Iresidual_regressor_residual_block_dense_6_biasadd_readvariableop_resource:Z
Hresidual_regressor_residual_block_dense_7_matmul_readvariableop_resource:W
Iresidual_regressor_residual_block_dense_7_biasadd_readvariableop_resource:\
Jresidual_regressor_residual_block_1_dense_8_matmul_readvariableop_resource:Y
Kresidual_regressor_residual_block_1_dense_8_biasadd_readvariableop_resource:\
Jresidual_regressor_residual_block_1_dense_9_matmul_readvariableop_resource:Y
Kresidual_regressor_residual_block_1_dense_9_biasadd_readvariableop_resource:L
:residual_regressor_dense_10_matmul_readvariableop_resource:I
;residual_regressor_dense_10_biasadd_readvariableop_resource:
identity¢2residual_regressor/dense_10/BiasAdd/ReadVariableOp¢1residual_regressor/dense_10/MatMul/ReadVariableOp¢1residual_regressor/dense_5/BiasAdd/ReadVariableOp¢0residual_regressor/dense_5/MatMul/ReadVariableOp¢@residual_regressor/residual_block/dense_6/BiasAdd/ReadVariableOp¢Bresidual_regressor/residual_block/dense_6/BiasAdd_1/ReadVariableOp¢Bresidual_regressor/residual_block/dense_6/BiasAdd_2/ReadVariableOp¢Bresidual_regressor/residual_block/dense_6/BiasAdd_3/ReadVariableOp¢?residual_regressor/residual_block/dense_6/MatMul/ReadVariableOp¢Aresidual_regressor/residual_block/dense_6/MatMul_1/ReadVariableOp¢Aresidual_regressor/residual_block/dense_6/MatMul_2/ReadVariableOp¢Aresidual_regressor/residual_block/dense_6/MatMul_3/ReadVariableOp¢@residual_regressor/residual_block/dense_7/BiasAdd/ReadVariableOp¢Bresidual_regressor/residual_block/dense_7/BiasAdd_1/ReadVariableOp¢Bresidual_regressor/residual_block/dense_7/BiasAdd_2/ReadVariableOp¢Bresidual_regressor/residual_block/dense_7/BiasAdd_3/ReadVariableOp¢?residual_regressor/residual_block/dense_7/MatMul/ReadVariableOp¢Aresidual_regressor/residual_block/dense_7/MatMul_1/ReadVariableOp¢Aresidual_regressor/residual_block/dense_7/MatMul_2/ReadVariableOp¢Aresidual_regressor/residual_block/dense_7/MatMul_3/ReadVariableOp¢Bresidual_regressor/residual_block_1/dense_8/BiasAdd/ReadVariableOp¢Aresidual_regressor/residual_block_1/dense_8/MatMul/ReadVariableOp¢Bresidual_regressor/residual_block_1/dense_9/BiasAdd/ReadVariableOp¢Aresidual_regressor/residual_block_1/dense_9/MatMul/ReadVariableOpÞ
0residual_regressor/dense_5/MatMul/ReadVariableOpReadVariableOp9residual_regressor_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0residual_regressor/dense_5/MatMul/ReadVariableOpÅ
!residual_regressor/dense_5/MatMulMatMulinput_18residual_regressor/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!residual_regressor/dense_5/MatMulÝ
1residual_regressor/dense_5/BiasAdd/ReadVariableOpReadVariableOp:residual_regressor_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1residual_regressor/dense_5/BiasAdd/ReadVariableOpí
"residual_regressor/dense_5/BiasAddBiasAdd+residual_regressor/dense_5/MatMul:product:09residual_regressor/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"residual_regressor/dense_5/BiasAdd¦
residual_regressor/dense_5/EluElu+residual_regressor/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
residual_regressor/dense_5/Elu
?residual_regressor/residual_block/dense_6/MatMul/ReadVariableOpReadVariableOpHresidual_regressor_residual_block_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02A
?residual_regressor/residual_block/dense_6/MatMul/ReadVariableOp
0residual_regressor/residual_block/dense_6/MatMulMatMul,residual_regressor/dense_5/Elu:activations:0Gresidual_regressor/residual_block/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0residual_regressor/residual_block/dense_6/MatMul
@residual_regressor/residual_block/dense_6/BiasAdd/ReadVariableOpReadVariableOpIresidual_regressor_residual_block_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@residual_regressor/residual_block/dense_6/BiasAdd/ReadVariableOp©
1residual_regressor/residual_block/dense_6/BiasAddBiasAdd:residual_regressor/residual_block/dense_6/MatMul:product:0Hresidual_regressor/residual_block/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1residual_regressor/residual_block/dense_6/BiasAddÓ
-residual_regressor/residual_block/dense_6/EluElu:residual_regressor/residual_block/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-residual_regressor/residual_block/dense_6/Elu
?residual_regressor/residual_block/dense_7/MatMul/ReadVariableOpReadVariableOpHresidual_regressor_residual_block_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02A
?residual_regressor/residual_block/dense_7/MatMul/ReadVariableOp¦
0residual_regressor/residual_block/dense_7/MatMulMatMul;residual_regressor/residual_block/dense_6/Elu:activations:0Gresidual_regressor/residual_block/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0residual_regressor/residual_block/dense_7/MatMul
@residual_regressor/residual_block/dense_7/BiasAdd/ReadVariableOpReadVariableOpIresidual_regressor_residual_block_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@residual_regressor/residual_block/dense_7/BiasAdd/ReadVariableOp©
1residual_regressor/residual_block/dense_7/BiasAddBiasAdd:residual_regressor/residual_block/dense_7/MatMul:product:0Hresidual_regressor/residual_block/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1residual_regressor/residual_block/dense_7/BiasAddÓ
-residual_regressor/residual_block/dense_7/EluElu:residual_regressor/residual_block/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-residual_regressor/residual_block/dense_7/Eluô
%residual_regressor/residual_block/addAddV2;residual_regressor/residual_block/dense_7/Elu:activations:0,residual_regressor/dense_5/Elu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%residual_regressor/residual_block/add
Aresidual_regressor/residual_block/dense_6/MatMul_1/ReadVariableOpReadVariableOpHresidual_regressor_residual_block_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
Aresidual_regressor/residual_block/dense_6/MatMul_1/ReadVariableOp
2residual_regressor/residual_block/dense_6/MatMul_1MatMul)residual_regressor/residual_block/add:z:0Iresidual_regressor/residual_block/dense_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2residual_regressor/residual_block/dense_6/MatMul_1
Bresidual_regressor/residual_block/dense_6/BiasAdd_1/ReadVariableOpReadVariableOpIresidual_regressor_residual_block_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bresidual_regressor/residual_block/dense_6/BiasAdd_1/ReadVariableOp±
3residual_regressor/residual_block/dense_6/BiasAdd_1BiasAdd<residual_regressor/residual_block/dense_6/MatMul_1:product:0Jresidual_regressor/residual_block/dense_6/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3residual_regressor/residual_block/dense_6/BiasAdd_1Ù
/residual_regressor/residual_block/dense_6/Elu_1Elu<residual_regressor/residual_block/dense_6/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/residual_regressor/residual_block/dense_6/Elu_1
Aresidual_regressor/residual_block/dense_7/MatMul_1/ReadVariableOpReadVariableOpHresidual_regressor_residual_block_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
Aresidual_regressor/residual_block/dense_7/MatMul_1/ReadVariableOp®
2residual_regressor/residual_block/dense_7/MatMul_1MatMul=residual_regressor/residual_block/dense_6/Elu_1:activations:0Iresidual_regressor/residual_block/dense_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2residual_regressor/residual_block/dense_7/MatMul_1
Bresidual_regressor/residual_block/dense_7/BiasAdd_1/ReadVariableOpReadVariableOpIresidual_regressor_residual_block_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bresidual_regressor/residual_block/dense_7/BiasAdd_1/ReadVariableOp±
3residual_regressor/residual_block/dense_7/BiasAdd_1BiasAdd<residual_regressor/residual_block/dense_7/MatMul_1:product:0Jresidual_regressor/residual_block/dense_7/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3residual_regressor/residual_block/dense_7/BiasAdd_1Ù
/residual_regressor/residual_block/dense_7/Elu_1Elu<residual_regressor/residual_block/dense_7/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/residual_regressor/residual_block/dense_7/Elu_1÷
'residual_regressor/residual_block/add_1AddV2=residual_regressor/residual_block/dense_7/Elu_1:activations:0)residual_regressor/residual_block/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'residual_regressor/residual_block/add_1
Aresidual_regressor/residual_block/dense_6/MatMul_2/ReadVariableOpReadVariableOpHresidual_regressor_residual_block_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
Aresidual_regressor/residual_block/dense_6/MatMul_2/ReadVariableOp
2residual_regressor/residual_block/dense_6/MatMul_2MatMul+residual_regressor/residual_block/add_1:z:0Iresidual_regressor/residual_block/dense_6/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2residual_regressor/residual_block/dense_6/MatMul_2
Bresidual_regressor/residual_block/dense_6/BiasAdd_2/ReadVariableOpReadVariableOpIresidual_regressor_residual_block_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bresidual_regressor/residual_block/dense_6/BiasAdd_2/ReadVariableOp±
3residual_regressor/residual_block/dense_6/BiasAdd_2BiasAdd<residual_regressor/residual_block/dense_6/MatMul_2:product:0Jresidual_regressor/residual_block/dense_6/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3residual_regressor/residual_block/dense_6/BiasAdd_2Ù
/residual_regressor/residual_block/dense_6/Elu_2Elu<residual_regressor/residual_block/dense_6/BiasAdd_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/residual_regressor/residual_block/dense_6/Elu_2
Aresidual_regressor/residual_block/dense_7/MatMul_2/ReadVariableOpReadVariableOpHresidual_regressor_residual_block_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
Aresidual_regressor/residual_block/dense_7/MatMul_2/ReadVariableOp®
2residual_regressor/residual_block/dense_7/MatMul_2MatMul=residual_regressor/residual_block/dense_6/Elu_2:activations:0Iresidual_regressor/residual_block/dense_7/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2residual_regressor/residual_block/dense_7/MatMul_2
Bresidual_regressor/residual_block/dense_7/BiasAdd_2/ReadVariableOpReadVariableOpIresidual_regressor_residual_block_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bresidual_regressor/residual_block/dense_7/BiasAdd_2/ReadVariableOp±
3residual_regressor/residual_block/dense_7/BiasAdd_2BiasAdd<residual_regressor/residual_block/dense_7/MatMul_2:product:0Jresidual_regressor/residual_block/dense_7/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3residual_regressor/residual_block/dense_7/BiasAdd_2Ù
/residual_regressor/residual_block/dense_7/Elu_2Elu<residual_regressor/residual_block/dense_7/BiasAdd_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/residual_regressor/residual_block/dense_7/Elu_2ù
'residual_regressor/residual_block/add_2AddV2=residual_regressor/residual_block/dense_7/Elu_2:activations:0+residual_regressor/residual_block/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'residual_regressor/residual_block/add_2
Aresidual_regressor/residual_block/dense_6/MatMul_3/ReadVariableOpReadVariableOpHresidual_regressor_residual_block_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
Aresidual_regressor/residual_block/dense_6/MatMul_3/ReadVariableOp
2residual_regressor/residual_block/dense_6/MatMul_3MatMul+residual_regressor/residual_block/add_2:z:0Iresidual_regressor/residual_block/dense_6/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2residual_regressor/residual_block/dense_6/MatMul_3
Bresidual_regressor/residual_block/dense_6/BiasAdd_3/ReadVariableOpReadVariableOpIresidual_regressor_residual_block_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bresidual_regressor/residual_block/dense_6/BiasAdd_3/ReadVariableOp±
3residual_regressor/residual_block/dense_6/BiasAdd_3BiasAdd<residual_regressor/residual_block/dense_6/MatMul_3:product:0Jresidual_regressor/residual_block/dense_6/BiasAdd_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3residual_regressor/residual_block/dense_6/BiasAdd_3Ù
/residual_regressor/residual_block/dense_6/Elu_3Elu<residual_regressor/residual_block/dense_6/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/residual_regressor/residual_block/dense_6/Elu_3
Aresidual_regressor/residual_block/dense_7/MatMul_3/ReadVariableOpReadVariableOpHresidual_regressor_residual_block_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
Aresidual_regressor/residual_block/dense_7/MatMul_3/ReadVariableOp®
2residual_regressor/residual_block/dense_7/MatMul_3MatMul=residual_regressor/residual_block/dense_6/Elu_3:activations:0Iresidual_regressor/residual_block/dense_7/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2residual_regressor/residual_block/dense_7/MatMul_3
Bresidual_regressor/residual_block/dense_7/BiasAdd_3/ReadVariableOpReadVariableOpIresidual_regressor_residual_block_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bresidual_regressor/residual_block/dense_7/BiasAdd_3/ReadVariableOp±
3residual_regressor/residual_block/dense_7/BiasAdd_3BiasAdd<residual_regressor/residual_block/dense_7/MatMul_3:product:0Jresidual_regressor/residual_block/dense_7/BiasAdd_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3residual_regressor/residual_block/dense_7/BiasAdd_3Ù
/residual_regressor/residual_block/dense_7/Elu_3Elu<residual_regressor/residual_block/dense_7/BiasAdd_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/residual_regressor/residual_block/dense_7/Elu_3ù
'residual_regressor/residual_block/add_3AddV2=residual_regressor/residual_block/dense_7/Elu_3:activations:0+residual_regressor/residual_block/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'residual_regressor/residual_block/add_3
Aresidual_regressor/residual_block_1/dense_8/MatMul/ReadVariableOpReadVariableOpJresidual_regressor_residual_block_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
Aresidual_regressor/residual_block_1/dense_8/MatMul/ReadVariableOp
2residual_regressor/residual_block_1/dense_8/MatMulMatMul+residual_regressor/residual_block/add_3:z:0Iresidual_regressor/residual_block_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2residual_regressor/residual_block_1/dense_8/MatMul
Bresidual_regressor/residual_block_1/dense_8/BiasAdd/ReadVariableOpReadVariableOpKresidual_regressor_residual_block_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bresidual_regressor/residual_block_1/dense_8/BiasAdd/ReadVariableOp±
3residual_regressor/residual_block_1/dense_8/BiasAddBiasAdd<residual_regressor/residual_block_1/dense_8/MatMul:product:0Jresidual_regressor/residual_block_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3residual_regressor/residual_block_1/dense_8/BiasAddÙ
/residual_regressor/residual_block_1/dense_8/EluElu<residual_regressor/residual_block_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/residual_regressor/residual_block_1/dense_8/Elu
Aresidual_regressor/residual_block_1/dense_9/MatMul/ReadVariableOpReadVariableOpJresidual_regressor_residual_block_1_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
Aresidual_regressor/residual_block_1/dense_9/MatMul/ReadVariableOp®
2residual_regressor/residual_block_1/dense_9/MatMulMatMul=residual_regressor/residual_block_1/dense_8/Elu:activations:0Iresidual_regressor/residual_block_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2residual_regressor/residual_block_1/dense_9/MatMul
Bresidual_regressor/residual_block_1/dense_9/BiasAdd/ReadVariableOpReadVariableOpKresidual_regressor_residual_block_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bresidual_regressor/residual_block_1/dense_9/BiasAdd/ReadVariableOp±
3residual_regressor/residual_block_1/dense_9/BiasAddBiasAdd<residual_regressor/residual_block_1/dense_9/MatMul:product:0Jresidual_regressor/residual_block_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3residual_regressor/residual_block_1/dense_9/BiasAddÙ
/residual_regressor/residual_block_1/dense_9/EluElu<residual_regressor/residual_block_1/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/residual_regressor/residual_block_1/dense_9/Eluù
'residual_regressor/residual_block_1/addAddV2=residual_regressor/residual_block_1/dense_9/Elu:activations:0+residual_regressor/residual_block/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'residual_regressor/residual_block_1/addá
1residual_regressor/dense_10/MatMul/ReadVariableOpReadVariableOp:residual_regressor_dense_10_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1residual_regressor/dense_10/MatMul/ReadVariableOpì
"residual_regressor/dense_10/MatMulMatMul+residual_regressor/residual_block_1/add:z:09residual_regressor/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"residual_regressor/dense_10/MatMulà
2residual_regressor/dense_10/BiasAdd/ReadVariableOpReadVariableOp;residual_regressor_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2residual_regressor/dense_10/BiasAdd/ReadVariableOpñ
#residual_regressor/dense_10/BiasAddBiasAdd,residual_regressor/dense_10/MatMul:product:0:residual_regressor/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#residual_regressor/dense_10/BiasAdd¢
IdentityIdentity,residual_regressor/dense_10/BiasAdd:output:03^residual_regressor/dense_10/BiasAdd/ReadVariableOp2^residual_regressor/dense_10/MatMul/ReadVariableOp2^residual_regressor/dense_5/BiasAdd/ReadVariableOp1^residual_regressor/dense_5/MatMul/ReadVariableOpA^residual_regressor/residual_block/dense_6/BiasAdd/ReadVariableOpC^residual_regressor/residual_block/dense_6/BiasAdd_1/ReadVariableOpC^residual_regressor/residual_block/dense_6/BiasAdd_2/ReadVariableOpC^residual_regressor/residual_block/dense_6/BiasAdd_3/ReadVariableOp@^residual_regressor/residual_block/dense_6/MatMul/ReadVariableOpB^residual_regressor/residual_block/dense_6/MatMul_1/ReadVariableOpB^residual_regressor/residual_block/dense_6/MatMul_2/ReadVariableOpB^residual_regressor/residual_block/dense_6/MatMul_3/ReadVariableOpA^residual_regressor/residual_block/dense_7/BiasAdd/ReadVariableOpC^residual_regressor/residual_block/dense_7/BiasAdd_1/ReadVariableOpC^residual_regressor/residual_block/dense_7/BiasAdd_2/ReadVariableOpC^residual_regressor/residual_block/dense_7/BiasAdd_3/ReadVariableOp@^residual_regressor/residual_block/dense_7/MatMul/ReadVariableOpB^residual_regressor/residual_block/dense_7/MatMul_1/ReadVariableOpB^residual_regressor/residual_block/dense_7/MatMul_2/ReadVariableOpB^residual_regressor/residual_block/dense_7/MatMul_3/ReadVariableOpC^residual_regressor/residual_block_1/dense_8/BiasAdd/ReadVariableOpB^residual_regressor/residual_block_1/dense_8/MatMul/ReadVariableOpC^residual_regressor/residual_block_1/dense_9/BiasAdd/ReadVariableOpB^residual_regressor/residual_block_1/dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2h
2residual_regressor/dense_10/BiasAdd/ReadVariableOp2residual_regressor/dense_10/BiasAdd/ReadVariableOp2f
1residual_regressor/dense_10/MatMul/ReadVariableOp1residual_regressor/dense_10/MatMul/ReadVariableOp2f
1residual_regressor/dense_5/BiasAdd/ReadVariableOp1residual_regressor/dense_5/BiasAdd/ReadVariableOp2d
0residual_regressor/dense_5/MatMul/ReadVariableOp0residual_regressor/dense_5/MatMul/ReadVariableOp2
@residual_regressor/residual_block/dense_6/BiasAdd/ReadVariableOp@residual_regressor/residual_block/dense_6/BiasAdd/ReadVariableOp2
Bresidual_regressor/residual_block/dense_6/BiasAdd_1/ReadVariableOpBresidual_regressor/residual_block/dense_6/BiasAdd_1/ReadVariableOp2
Bresidual_regressor/residual_block/dense_6/BiasAdd_2/ReadVariableOpBresidual_regressor/residual_block/dense_6/BiasAdd_2/ReadVariableOp2
Bresidual_regressor/residual_block/dense_6/BiasAdd_3/ReadVariableOpBresidual_regressor/residual_block/dense_6/BiasAdd_3/ReadVariableOp2
?residual_regressor/residual_block/dense_6/MatMul/ReadVariableOp?residual_regressor/residual_block/dense_6/MatMul/ReadVariableOp2
Aresidual_regressor/residual_block/dense_6/MatMul_1/ReadVariableOpAresidual_regressor/residual_block/dense_6/MatMul_1/ReadVariableOp2
Aresidual_regressor/residual_block/dense_6/MatMul_2/ReadVariableOpAresidual_regressor/residual_block/dense_6/MatMul_2/ReadVariableOp2
Aresidual_regressor/residual_block/dense_6/MatMul_3/ReadVariableOpAresidual_regressor/residual_block/dense_6/MatMul_3/ReadVariableOp2
@residual_regressor/residual_block/dense_7/BiasAdd/ReadVariableOp@residual_regressor/residual_block/dense_7/BiasAdd/ReadVariableOp2
Bresidual_regressor/residual_block/dense_7/BiasAdd_1/ReadVariableOpBresidual_regressor/residual_block/dense_7/BiasAdd_1/ReadVariableOp2
Bresidual_regressor/residual_block/dense_7/BiasAdd_2/ReadVariableOpBresidual_regressor/residual_block/dense_7/BiasAdd_2/ReadVariableOp2
Bresidual_regressor/residual_block/dense_7/BiasAdd_3/ReadVariableOpBresidual_regressor/residual_block/dense_7/BiasAdd_3/ReadVariableOp2
?residual_regressor/residual_block/dense_7/MatMul/ReadVariableOp?residual_regressor/residual_block/dense_7/MatMul/ReadVariableOp2
Aresidual_regressor/residual_block/dense_7/MatMul_1/ReadVariableOpAresidual_regressor/residual_block/dense_7/MatMul_1/ReadVariableOp2
Aresidual_regressor/residual_block/dense_7/MatMul_2/ReadVariableOpAresidual_regressor/residual_block/dense_7/MatMul_2/ReadVariableOp2
Aresidual_regressor/residual_block/dense_7/MatMul_3/ReadVariableOpAresidual_regressor/residual_block/dense_7/MatMul_3/ReadVariableOp2
Bresidual_regressor/residual_block_1/dense_8/BiasAdd/ReadVariableOpBresidual_regressor/residual_block_1/dense_8/BiasAdd/ReadVariableOp2
Aresidual_regressor/residual_block_1/dense_8/MatMul/ReadVariableOpAresidual_regressor/residual_block_1/dense_8/MatMul/ReadVariableOp2
Bresidual_regressor/residual_block_1/dense_9/BiasAdd/ReadVariableOpBresidual_regressor/residual_block_1/dense_9/BiasAdd/ReadVariableOp2
Aresidual_regressor/residual_block_1/dense_9/MatMul/ReadVariableOpAresidual_regressor/residual_block_1/dense_9/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¦

ó
B__inference_dense_5_layer_call_and_return_conditional_losses_97495

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elu
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
+
½
M__inference_residual_regressor_layer_call_and_return_conditional_losses_97360
input_1
dense_5_97265:
dense_5_97267:&
residual_block_97290:"
residual_block_97292:&
residual_block_97294:"
residual_block_97296:(
residual_block_1_97334:$
residual_block_1_97336:(
residual_block_1_97338:$
residual_block_1_97340: 
dense_10_97354:
dense_10_97356:
identity¢ dense_10/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢&residual_block/StatefulPartitionedCall¢(residual_block/StatefulPartitionedCall_1¢(residual_block/StatefulPartitionedCall_2¢(residual_block/StatefulPartitionedCall_3¢(residual_block_1/StatefulPartitionedCall
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_5_97265dense_5_97267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_972642!
dense_5/StatefulPartitionedCall
&residual_block/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0residual_block_97290residual_block_97292residual_block_97294residual_block_97296*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_block_layer_call_and_return_conditional_losses_972892(
&residual_block/StatefulPartitionedCall
(residual_block/StatefulPartitionedCall_1StatefulPartitionedCall/residual_block/StatefulPartitionedCall:output:0residual_block_97290residual_block_97292residual_block_97294residual_block_97296*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_block_layer_call_and_return_conditional_losses_972892*
(residual_block/StatefulPartitionedCall_1
(residual_block/StatefulPartitionedCall_2StatefulPartitionedCall1residual_block/StatefulPartitionedCall_1:output:0residual_block_97290residual_block_97292residual_block_97294residual_block_97296*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_block_layer_call_and_return_conditional_losses_972892*
(residual_block/StatefulPartitionedCall_2
(residual_block/StatefulPartitionedCall_3StatefulPartitionedCall1residual_block/StatefulPartitionedCall_2:output:0residual_block_97290residual_block_97292residual_block_97294residual_block_97296*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_block_layer_call_and_return_conditional_losses_972892*
(residual_block/StatefulPartitionedCall_3
(residual_block_1/StatefulPartitionedCallStatefulPartitionedCall1residual_block/StatefulPartitionedCall_3:output:0residual_block_1_97334residual_block_1_97336residual_block_1_97338residual_block_1_97340*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_residual_block_1_layer_call_and_return_conditional_losses_973332*
(residual_block_1/StatefulPartitionedCall¼
 dense_10/StatefulPartitionedCallStatefulPartitionedCall1residual_block_1/StatefulPartitionedCall:output:0dense_10_97354dense_10_97356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_973532"
 dense_10/StatefulPartitionedCall
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0!^dense_10/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall'^residual_block/StatefulPartitionedCall)^residual_block/StatefulPartitionedCall_1)^residual_block/StatefulPartitionedCall_2)^residual_block/StatefulPartitionedCall_3)^residual_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2P
&residual_block/StatefulPartitionedCall&residual_block/StatefulPartitionedCall2T
(residual_block/StatefulPartitionedCall_1(residual_block/StatefulPartitionedCall_12T
(residual_block/StatefulPartitionedCall_2(residual_block/StatefulPartitionedCall_22T
(residual_block/StatefulPartitionedCall_3(residual_block/StatefulPartitionedCall_32T
(residual_block_1/StatefulPartitionedCall(residual_block_1/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ü
Î
K__inference_residual_block_1_layer_call_and_return_conditional_losses_97559

inputs8
&dense_8_matmul_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¥
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/BiasAddm
dense_8/EluEludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/Elu¥
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMuldense_8/Elu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp¡
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/BiasAddm
dense_9/EluEludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/Eluh
addAddV2dense_9/Elu:activations:0inputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addÝ
IdentityIdentityadd:z:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ßi
»
__inference__traced_save_97733
file_prefix@
<savev2_residual_regressor_dense_5_kernel_read_readvariableop>
:savev2_residual_regressor_dense_5_bias_read_readvariableopA
=savev2_residual_regressor_dense_10_kernel_read_readvariableop?
;savev2_residual_regressor_dense_10_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableopO
Ksavev2_residual_regressor_residual_block_dense_6_kernel_read_readvariableopM
Isavev2_residual_regressor_residual_block_dense_6_bias_read_readvariableopO
Ksavev2_residual_regressor_residual_block_dense_7_kernel_read_readvariableopM
Isavev2_residual_regressor_residual_block_dense_7_bias_read_readvariableopQ
Msavev2_residual_regressor_residual_block_1_dense_8_kernel_read_readvariableopO
Ksavev2_residual_regressor_residual_block_1_dense_8_bias_read_readvariableopQ
Msavev2_residual_regressor_residual_block_1_dense_9_kernel_read_readvariableopO
Ksavev2_residual_regressor_residual_block_1_dense_9_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopH
Dsavev2_nadam_residual_regressor_dense_5_kernel_m_read_readvariableopF
Bsavev2_nadam_residual_regressor_dense_5_bias_m_read_readvariableopI
Esavev2_nadam_residual_regressor_dense_10_kernel_m_read_readvariableopG
Csavev2_nadam_residual_regressor_dense_10_bias_m_read_readvariableopW
Ssavev2_nadam_residual_regressor_residual_block_dense_6_kernel_m_read_readvariableopU
Qsavev2_nadam_residual_regressor_residual_block_dense_6_bias_m_read_readvariableopW
Ssavev2_nadam_residual_regressor_residual_block_dense_7_kernel_m_read_readvariableopU
Qsavev2_nadam_residual_regressor_residual_block_dense_7_bias_m_read_readvariableopY
Usavev2_nadam_residual_regressor_residual_block_1_dense_8_kernel_m_read_readvariableopW
Ssavev2_nadam_residual_regressor_residual_block_1_dense_8_bias_m_read_readvariableopY
Usavev2_nadam_residual_regressor_residual_block_1_dense_9_kernel_m_read_readvariableopW
Ssavev2_nadam_residual_regressor_residual_block_1_dense_9_bias_m_read_readvariableopH
Dsavev2_nadam_residual_regressor_dense_5_kernel_v_read_readvariableopF
Bsavev2_nadam_residual_regressor_dense_5_bias_v_read_readvariableopI
Esavev2_nadam_residual_regressor_dense_10_kernel_v_read_readvariableopG
Csavev2_nadam_residual_regressor_dense_10_bias_v_read_readvariableopW
Ssavev2_nadam_residual_regressor_residual_block_dense_6_kernel_v_read_readvariableopU
Qsavev2_nadam_residual_regressor_residual_block_dense_6_bias_v_read_readvariableopW
Ssavev2_nadam_residual_regressor_residual_block_dense_7_kernel_v_read_readvariableopU
Qsavev2_nadam_residual_regressor_residual_block_dense_7_bias_v_read_readvariableopY
Usavev2_nadam_residual_regressor_residual_block_1_dense_8_kernel_v_read_readvariableopW
Ssavev2_nadam_residual_regressor_residual_block_1_dense_8_bias_v_read_readvariableopY
Usavev2_nadam_residual_regressor_residual_block_1_dense_9_kernel_v_read_readvariableopW
Ssavev2_nadam_residual_regressor_residual_block_1_dense_9_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*
valueB-B(hidden/kernel/.ATTRIBUTES/VARIABLE_VALUEB&hidden/bias/.ATTRIBUTES/VARIABLE_VALUEB-outputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB+outputLayer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBDhidden/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBhidden/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIoutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGoutputLayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDhidden/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBhidden/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIoutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGoutputLayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesâ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_residual_regressor_dense_5_kernel_read_readvariableop:savev2_residual_regressor_dense_5_bias_read_readvariableop=savev2_residual_regressor_dense_10_kernel_read_readvariableop;savev2_residual_regressor_dense_10_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableopKsavev2_residual_regressor_residual_block_dense_6_kernel_read_readvariableopIsavev2_residual_regressor_residual_block_dense_6_bias_read_readvariableopKsavev2_residual_regressor_residual_block_dense_7_kernel_read_readvariableopIsavev2_residual_regressor_residual_block_dense_7_bias_read_readvariableopMsavev2_residual_regressor_residual_block_1_dense_8_kernel_read_readvariableopKsavev2_residual_regressor_residual_block_1_dense_8_bias_read_readvariableopMsavev2_residual_regressor_residual_block_1_dense_9_kernel_read_readvariableopKsavev2_residual_regressor_residual_block_1_dense_9_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopDsavev2_nadam_residual_regressor_dense_5_kernel_m_read_readvariableopBsavev2_nadam_residual_regressor_dense_5_bias_m_read_readvariableopEsavev2_nadam_residual_regressor_dense_10_kernel_m_read_readvariableopCsavev2_nadam_residual_regressor_dense_10_bias_m_read_readvariableopSsavev2_nadam_residual_regressor_residual_block_dense_6_kernel_m_read_readvariableopQsavev2_nadam_residual_regressor_residual_block_dense_6_bias_m_read_readvariableopSsavev2_nadam_residual_regressor_residual_block_dense_7_kernel_m_read_readvariableopQsavev2_nadam_residual_regressor_residual_block_dense_7_bias_m_read_readvariableopUsavev2_nadam_residual_regressor_residual_block_1_dense_8_kernel_m_read_readvariableopSsavev2_nadam_residual_regressor_residual_block_1_dense_8_bias_m_read_readvariableopUsavev2_nadam_residual_regressor_residual_block_1_dense_9_kernel_m_read_readvariableopSsavev2_nadam_residual_regressor_residual_block_1_dense_9_bias_m_read_readvariableopDsavev2_nadam_residual_regressor_dense_5_kernel_v_read_readvariableopBsavev2_nadam_residual_regressor_dense_5_bias_v_read_readvariableopEsavev2_nadam_residual_regressor_dense_10_kernel_v_read_readvariableopCsavev2_nadam_residual_regressor_dense_10_bias_v_read_readvariableopSsavev2_nadam_residual_regressor_residual_block_dense_6_kernel_v_read_readvariableopQsavev2_nadam_residual_regressor_residual_block_dense_6_bias_v_read_readvariableopSsavev2_nadam_residual_regressor_residual_block_dense_7_kernel_v_read_readvariableopQsavev2_nadam_residual_regressor_residual_block_dense_7_bias_v_read_readvariableopUsavev2_nadam_residual_regressor_residual_block_1_dense_8_kernel_v_read_readvariableopSsavev2_nadam_residual_regressor_residual_block_1_dense_8_bias_v_read_readvariableopUsavev2_nadam_residual_regressor_residual_block_1_dense_9_kernel_v_read_readvariableopSsavev2_nadam_residual_regressor_residual_block_1_dense_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *;
dtypes1
/2-	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*É
_input_shapes·
´: ::::: : : : : : ::::::::: : ::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

:: *

_output_shapes
::$+ 

_output_shapes

:: ,

_output_shapes
::-

_output_shapes
: 
Ï	
ô
C__inference_dense_10_layer_call_and_return_conditional_losses_97353

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


 
#__inference_signature_wrapper_97475
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_972492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¦

ó
B__inference_dense_5_layer_call_and_return_conditional_losses_97264

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elu
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
Ó
0__inference_residual_block_1_layer_call_fn_97540

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_residual_block_1_layer_call_and_return_conditional_losses_973332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µË
ô"
!__inference__traced_restore_97875
file_prefixD
2assignvariableop_residual_regressor_dense_5_kernel:@
2assignvariableop_1_residual_regressor_dense_5_bias:G
5assignvariableop_2_residual_regressor_dense_10_kernel:A
3assignvariableop_3_residual_regressor_dense_10_bias:'
assignvariableop_4_nadam_iter:	 )
assignvariableop_5_nadam_beta_1: )
assignvariableop_6_nadam_beta_2: (
assignvariableop_7_nadam_decay: 0
&assignvariableop_8_nadam_learning_rate: 1
'assignvariableop_9_nadam_momentum_cache: V
Dassignvariableop_10_residual_regressor_residual_block_dense_6_kernel:P
Bassignvariableop_11_residual_regressor_residual_block_dense_6_bias:V
Dassignvariableop_12_residual_regressor_residual_block_dense_7_kernel:P
Bassignvariableop_13_residual_regressor_residual_block_dense_7_bias:X
Fassignvariableop_14_residual_regressor_residual_block_1_dense_8_kernel:R
Dassignvariableop_15_residual_regressor_residual_block_1_dense_8_bias:X
Fassignvariableop_16_residual_regressor_residual_block_1_dense_9_kernel:R
Dassignvariableop_17_residual_regressor_residual_block_1_dense_9_bias:#
assignvariableop_18_total: #
assignvariableop_19_count: O
=assignvariableop_20_nadam_residual_regressor_dense_5_kernel_m:I
;assignvariableop_21_nadam_residual_regressor_dense_5_bias_m:P
>assignvariableop_22_nadam_residual_regressor_dense_10_kernel_m:J
<assignvariableop_23_nadam_residual_regressor_dense_10_bias_m:^
Lassignvariableop_24_nadam_residual_regressor_residual_block_dense_6_kernel_m:X
Jassignvariableop_25_nadam_residual_regressor_residual_block_dense_6_bias_m:^
Lassignvariableop_26_nadam_residual_regressor_residual_block_dense_7_kernel_m:X
Jassignvariableop_27_nadam_residual_regressor_residual_block_dense_7_bias_m:`
Nassignvariableop_28_nadam_residual_regressor_residual_block_1_dense_8_kernel_m:Z
Lassignvariableop_29_nadam_residual_regressor_residual_block_1_dense_8_bias_m:`
Nassignvariableop_30_nadam_residual_regressor_residual_block_1_dense_9_kernel_m:Z
Lassignvariableop_31_nadam_residual_regressor_residual_block_1_dense_9_bias_m:O
=assignvariableop_32_nadam_residual_regressor_dense_5_kernel_v:I
;assignvariableop_33_nadam_residual_regressor_dense_5_bias_v:P
>assignvariableop_34_nadam_residual_regressor_dense_10_kernel_v:J
<assignvariableop_35_nadam_residual_regressor_dense_10_bias_v:^
Lassignvariableop_36_nadam_residual_regressor_residual_block_dense_6_kernel_v:X
Jassignvariableop_37_nadam_residual_regressor_residual_block_dense_6_bias_v:^
Lassignvariableop_38_nadam_residual_regressor_residual_block_dense_7_kernel_v:X
Jassignvariableop_39_nadam_residual_regressor_residual_block_dense_7_bias_v:`
Nassignvariableop_40_nadam_residual_regressor_residual_block_1_dense_8_kernel_v:Z
Lassignvariableop_41_nadam_residual_regressor_residual_block_1_dense_8_bias_v:`
Nassignvariableop_42_nadam_residual_regressor_residual_block_1_dense_9_kernel_v:Z
Lassignvariableop_43_nadam_residual_regressor_residual_block_1_dense_9_bias_v:
identity_45¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*
valueB-B(hidden/kernel/.ATTRIBUTES/VARIABLE_VALUEB&hidden/bias/.ATTRIBUTES/VARIABLE_VALUEB-outputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB+outputLayer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBDhidden/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBhidden/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIoutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGoutputLayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDhidden/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBhidden/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIoutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGoutputLayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesè
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ê
_output_shapes·
´:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity±
AssignVariableOpAssignVariableOp2assignvariableop_residual_regressor_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1·
AssignVariableOp_1AssignVariableOp2assignvariableop_1_residual_regressor_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2º
AssignVariableOp_2AssignVariableOp5assignvariableop_2_residual_regressor_dense_10_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¸
AssignVariableOp_3AssignVariableOp3assignvariableop_3_residual_regressor_dense_10_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4¢
AssignVariableOp_4AssignVariableOpassignvariableop_4_nadam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¤
AssignVariableOp_5AssignVariableOpassignvariableop_5_nadam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_nadam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_nadam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8«
AssignVariableOp_8AssignVariableOp&assignvariableop_8_nadam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¬
AssignVariableOp_9AssignVariableOp'assignvariableop_9_nadam_momentum_cacheIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ì
AssignVariableOp_10AssignVariableOpDassignvariableop_10_residual_regressor_residual_block_dense_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ê
AssignVariableOp_11AssignVariableOpBassignvariableop_11_residual_regressor_residual_block_dense_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ì
AssignVariableOp_12AssignVariableOpDassignvariableop_12_residual_regressor_residual_block_dense_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ê
AssignVariableOp_13AssignVariableOpBassignvariableop_13_residual_regressor_residual_block_dense_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_residual_regressor_residual_block_1_dense_8_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ì
AssignVariableOp_15AssignVariableOpDassignvariableop_15_residual_regressor_residual_block_1_dense_8_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Î
AssignVariableOp_16AssignVariableOpFassignvariableop_16_residual_regressor_residual_block_1_dense_9_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ì
AssignVariableOp_17AssignVariableOpDassignvariableop_17_residual_regressor_residual_block_1_dense_9_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¡
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Å
AssignVariableOp_20AssignVariableOp=assignvariableop_20_nadam_residual_regressor_dense_5_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ã
AssignVariableOp_21AssignVariableOp;assignvariableop_21_nadam_residual_regressor_dense_5_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Æ
AssignVariableOp_22AssignVariableOp>assignvariableop_22_nadam_residual_regressor_dense_10_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ä
AssignVariableOp_23AssignVariableOp<assignvariableop_23_nadam_residual_regressor_dense_10_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ô
AssignVariableOp_24AssignVariableOpLassignvariableop_24_nadam_residual_regressor_residual_block_dense_6_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ò
AssignVariableOp_25AssignVariableOpJassignvariableop_25_nadam_residual_regressor_residual_block_dense_6_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ô
AssignVariableOp_26AssignVariableOpLassignvariableop_26_nadam_residual_regressor_residual_block_dense_7_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ò
AssignVariableOp_27AssignVariableOpJassignvariableop_27_nadam_residual_regressor_residual_block_dense_7_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ö
AssignVariableOp_28AssignVariableOpNassignvariableop_28_nadam_residual_regressor_residual_block_1_dense_8_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ô
AssignVariableOp_29AssignVariableOpLassignvariableop_29_nadam_residual_regressor_residual_block_1_dense_8_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ö
AssignVariableOp_30AssignVariableOpNassignvariableop_30_nadam_residual_regressor_residual_block_1_dense_9_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ô
AssignVariableOp_31AssignVariableOpLassignvariableop_31_nadam_residual_regressor_residual_block_1_dense_9_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Å
AssignVariableOp_32AssignVariableOp=assignvariableop_32_nadam_residual_regressor_dense_5_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ã
AssignVariableOp_33AssignVariableOp;assignvariableop_33_nadam_residual_regressor_dense_5_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Æ
AssignVariableOp_34AssignVariableOp>assignvariableop_34_nadam_residual_regressor_dense_10_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ä
AssignVariableOp_35AssignVariableOp<assignvariableop_35_nadam_residual_regressor_dense_10_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ô
AssignVariableOp_36AssignVariableOpLassignvariableop_36_nadam_residual_regressor_residual_block_dense_6_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ò
AssignVariableOp_37AssignVariableOpJassignvariableop_37_nadam_residual_regressor_residual_block_dense_6_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ô
AssignVariableOp_38AssignVariableOpLassignvariableop_38_nadam_residual_regressor_residual_block_dense_7_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ò
AssignVariableOp_39AssignVariableOpJassignvariableop_39_nadam_residual_regressor_residual_block_dense_7_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ö
AssignVariableOp_40AssignVariableOpNassignvariableop_40_nadam_residual_regressor_residual_block_1_dense_8_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ô
AssignVariableOp_41AssignVariableOpLassignvariableop_41_nadam_residual_regressor_residual_block_1_dense_8_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ö
AssignVariableOp_42AssignVariableOpNassignvariableop_42_nadam_residual_regressor_residual_block_1_dense_9_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ô
AssignVariableOp_43AssignVariableOpLassignvariableop_43_nadam_residual_regressor_residual_block_1_dense_9_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_439
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¦
Identity_44Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_44
Identity_45IdentityIdentity_44:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_45"#
identity_45Identity_45:output:0*m
_input_shapes\
Z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ù
Ñ
.__inference_residual_block_layer_call_fn_97508

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_residual_block_layer_call_and_return_conditional_losses_972892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ð¹
	

hidden
ResidualBlock_1
ResidualBlock_2
outputLayer
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
+&call_and_return_all_conditional_losses
__call__
_default_save_signature"ù
_tf_keras_modelß{"name": "residual_regressor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "ResidualRegressor", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 8]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "ResidualRegressor"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.009999999776482582, "decay": 0.004, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
Æ

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 30, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
»

hidden
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"name": "residual_block", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ResidualBlock", "config": {"layer was saved without config": true}}
½

hidden
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses" 
_tf_keras_layer{"name": "residual_block_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ResidualBlock", "config": {"layer was saved without config": true}}
Ñ

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
__call__
+&call_and_return_all_conditional_losses"ª
_tf_keras_layer{"name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
Ì
!iter

"beta_1

#beta_2
	$decay
%learning_rate
&momentum_cachemumvmwmx'my(mz)m{*m|+m},m~-m.mvvvv'v(v)v*v+v,v-v.v"
	optimizer
v
0
1
'2
(3
)4
*5
+6
,7
-8
.9
10
11"
trackable_list_wrapper
v
0
1
'2
(3
)4
*5
+6
,7
-8
.9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
trainable_variables
	variables
/non_trainable_variables

0layers
1layer_metrics
2metrics
regularization_losses
3layer_regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
3:12!residual_regressor/dense_5/kernel
-:+2residual_regressor/dense_5/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
	variables
4non_trainable_variables

5layers
6layer_metrics
7metrics
regularization_losses
8layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
90
:1"
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
	variables
;non_trainable_variables

<layers
=layer_metrics
>metrics
regularization_losses
?layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
<
+0
,1
-2
.3"
trackable_list_wrapper
<
+0
,1
-2
.3"
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
	variables
Bnon_trainable_variables

Clayers
Dlayer_metrics
Emetrics
regularization_losses
Flayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
4:22"residual_regressor/dense_10/kernel
.:,2 residual_regressor/dense_10/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
	variables
Gnon_trainable_variables

Hlayers
Ilayer_metrics
Jmetrics
regularization_losses
Klayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
B:@20residual_regressor/residual_block/dense_6/kernel
<::2.residual_regressor/residual_block/dense_6/bias
B:@20residual_regressor/residual_block/dense_7/kernel
<::2.residual_regressor/residual_block/dense_7/bias
D:B22residual_regressor/residual_block_1/dense_8/kernel
>:<20residual_regressor/residual_block_1/dense_8/bias
D:B22residual_regressor/residual_block_1/dense_9/kernel
>:<20residual_regressor/residual_block_1/dense_9/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ê

'kernel
(bias
Mtrainable_variables
N	variables
Oregularization_losses
P	keras_api
__call__
+&call_and_return_all_conditional_losses"£
_tf_keras_layer{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 30, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
Ì

)kernel
*bias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
__call__
+&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 30, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ì

+kernel
,bias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
__call__
+&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 30, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
Ì

-kernel
.bias
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
__call__
+ &call_and_return_all_conditional_losses"¥
_tf_keras_layer{"name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 30, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ô
	]total
	^count
_	variables
`	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 24}
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Mtrainable_variables
N	variables
anon_trainable_variables

blayers
clayer_metrics
dmetrics
Oregularization_losses
elayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Qtrainable_variables
R	variables
fnon_trainable_variables

glayers
hlayer_metrics
imetrics
Sregularization_losses
jlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Utrainable_variables
V	variables
knon_trainable_variables

llayers
mlayer_metrics
nmetrics
Wregularization_losses
olayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Ytrainable_variables
Z	variables
pnon_trainable_variables

qlayers
rlayer_metrics
smetrics
[regularization_losses
tlayer_regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
.
]0
^1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
9:72)Nadam/residual_regressor/dense_5/kernel/m
3:12'Nadam/residual_regressor/dense_5/bias/m
::82*Nadam/residual_regressor/dense_10/kernel/m
4:22(Nadam/residual_regressor/dense_10/bias/m
H:F28Nadam/residual_regressor/residual_block/dense_6/kernel/m
B:@26Nadam/residual_regressor/residual_block/dense_6/bias/m
H:F28Nadam/residual_regressor/residual_block/dense_7/kernel/m
B:@26Nadam/residual_regressor/residual_block/dense_7/bias/m
J:H2:Nadam/residual_regressor/residual_block_1/dense_8/kernel/m
D:B28Nadam/residual_regressor/residual_block_1/dense_8/bias/m
J:H2:Nadam/residual_regressor/residual_block_1/dense_9/kernel/m
D:B28Nadam/residual_regressor/residual_block_1/dense_9/bias/m
9:72)Nadam/residual_regressor/dense_5/kernel/v
3:12'Nadam/residual_regressor/dense_5/bias/v
::82*Nadam/residual_regressor/dense_10/kernel/v
4:22(Nadam/residual_regressor/dense_10/bias/v
H:F28Nadam/residual_regressor/residual_block/dense_6/kernel/v
B:@26Nadam/residual_regressor/residual_block/dense_6/bias/v
H:F28Nadam/residual_regressor/residual_block/dense_7/kernel/v
B:@26Nadam/residual_regressor/residual_block/dense_7/bias/v
J:H2:Nadam/residual_regressor/residual_block_1/dense_8/kernel/v
D:B28Nadam/residual_regressor/residual_block_1/dense_8/bias/v
J:H2:Nadam/residual_regressor/residual_block_1/dense_9/kernel/v
D:B28Nadam/residual_regressor/residual_block_1/dense_9/bias/v
2
M__inference_residual_regressor_layer_call_and_return_conditional_losses_97360Æ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
2ý
2__inference_residual_regressor_layer_call_fn_97390Æ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
Þ2Û
 __inference__wrapped_model_97249¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
Ñ2Î
'__inference_dense_5_layer_call_fn_97484¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_5_layer_call_and_return_conditional_losses_97495¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_residual_block_layer_call_fn_97508¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_residual_block_layer_call_and_return_conditional_losses_97527¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_residual_block_1_layer_call_fn_97540¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_residual_block_1_layer_call_and_return_conditional_losses_97559¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_10_layer_call_fn_97568¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_10_layer_call_and_return_conditional_losses_97578¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÊBÇ
#__inference_signature_wrapper_97475input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 __inference__wrapped_model_97249u'()*+,-.0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_10_layer_call_and_return_conditional_losses_97578\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_10_layer_call_fn_97568O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_5_layer_call_and_return_conditional_losses_97495\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_5_layer_call_fn_97484O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ­
K__inference_residual_block_1_layer_call_and_return_conditional_losses_97559^+,-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_residual_block_1_layer_call_fn_97540Q+,-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_residual_block_layer_call_and_return_conditional_losses_97527^'()*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_residual_block_layer_call_fn_97508Q'()*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¸
M__inference_residual_regressor_layer_call_and_return_conditional_losses_97360g'()*+,-.0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
2__inference_residual_regressor_layer_call_fn_97390Z'()*+,-.0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
#__inference_signature_wrapper_97475'()*+,-.;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ