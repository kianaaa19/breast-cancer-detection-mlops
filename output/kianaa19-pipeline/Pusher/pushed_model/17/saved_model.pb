ԏ
�.�-
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( ""
Ttype:
2	"
Tidxtype0:
2	
$
DisableCopyOnRead
resource�
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
�
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype�
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype�
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype�
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
�
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint���������
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.12v2.13.0-17-gf841394b1b78��
��
ConstConst*
_output_shapes	
:�*
dtype0	*ğ
value��B��	�"��                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      
��
Const_1Const*
_output_shapes	
:�*
dtype0*�
valueܚBؚ�BtoBiByouBaBtheBuBinBmeBandBforBmyBisByourBofBhaveBitBcallBimBthatBareBonBdoBsoBwithBorBnowBwillBifBgetBbutBbeBupBjustBthisBnotBatBnoBcanBwhatBurBhowBwhenBitsBknowB2BweBokBfromBillBdontBgotBtextBsendBneedBgoBloveBgoingBfreeBallB4BthenBoutBlikeBheBwantBdayBwasBamBourBcantBbeenBbackBcomeBtodayBtakeBsorryBgoodBasBthereBsomeBwayBtimeBthinkBohBnBhiBdidBdearBtxtBstopBstillBonlyBlorBreplyBnewBltgtBlaterBhomeBherBdaBworkBveryBrBpleaseBphoneBhadBdoingBbyBbabeBalreadyBwellBtheyBtellBmobileBhereByesBwonBwatBthatsBseeBnightBfindBbBanyBanBtooBsureBmsgBmoreBmissBhelpBhasBdBÌByetByeahBwhereB	somethingBshouldBreallyBplsBnokiaBhelloBgiveBcosBafterBwouldBwhyBweekBwaitBsayBrightBprizeBplaceBpickBoneBmorningBmakeBlongBlifeBlastBfewBdoneBcareBwhoBwhichBwereBsoonBsexyBnothingBheyBfriendsBeveryBcontactBclaimBbestBbeingBampBaboutB16BurgentBtryingBthanksBsomeoneBsheBsentBsaidBmoneyBlolBletBkeepBhaventBeBdownBbeforeBalwaysBaccountBabtB3B1BwinBwenBwaitingBuseBthingsBthemBthanBpartBoverBmuchBmessageBhopeBhimBhappyBgreatBfuckBfeelBeverBdidntBclassBcheckBcarBaskBanythingByupByearsBwontBwishBtomorrowBtoldBsmsBsmokeBsameBquiteBprincessBoffBniceBnameBmeetBmaybeBmayBliveBlatestBhahaBgudB
guaranteedBemailBdunBdoubleBcustomerBcashBcBbuyBalsoBå£1000BÌÏByaBwhatsBwentBweeksBwatchBusBtvBtryBtrueBtonightBthkBthingBthanxBtestBshowsBsatBsBrallyBpriceBplzBplanBpicsBpicBnumberBneverBmostBminsBluvBlightBleaveBlarBkBiveBidBhowsBhmvBhalfBguessBgoesBgettingB
everythingBdinnerBdaysBchanceBcameraBboredBbookBangryB5ByourselfByoureByBwkBurselfBtodaysBtmrBthoughBthoseBtextsBsirBsinceBservicesBserviceBsellBselectedBroomBrememberBreceiveBpointsBpersonBperBpeopleBpcBonceBofferBniteBnextBmomBmindBmeetingBmeansBlookBlineBlateBknwBjobBhourBhisBhesBhearBhaBgonnaBfirstBfinishBfineBeveningBendBelseBdatBcouldBcostBcoolBchatBchangeBcameBbecauseB	beautifulBaroundBanymoreBanotherBaightBahBagainB2ndBå£2000Bå£150ByearBwilBwidBwantedBwanBwalkingBvideoBvBunsubscribeB
unredeemedBtwoBtriedBtownBtopBtoneBthoughtBtheresBtheirBthankBtcsBtalkBtakesBsweetBstuffB	statementBspecialBsisBshoppingBshitBsexBsawBsavamobBrunBreadyBrateBprivateBpointBplusBparkBownBosoBoffersBnetworkBmovieBmonthBminutesBmanBmakingBmBlovableBlotsBloginBlittleBletsBlandlineBkissBjusBjoysBjohnB
identifierBhouseBhellBhappenedBhandBhafBguysBgamesBfunBfrndBfatherBfamilyBextraBexpiresBeitherBeasyBdunnoBdudeBdropBdoesBdeliveryBcongratsBcomesBcodeBcallingBcalledBbslvylBbrightBbetweenBbetterBb4BawardBattemptBardBapplyBaintB	afternoonBableB800B25pB2003B1stB08000930705B‰ÛÒBå£5000Bå£500Bå£3wkByoByestBxmasBxBwwwcomuknetBwrcBworryBworldBworkingBwordsBwordBwonderBwokeBwitBwinnerBwhileBweekendBwatchingBwakeBvouchersBvoucherBuveBusedBuntilBtscsBtreatBtoughBtonesBtogetherBthnkBthinkingBtheseBteachBstayBstartB	speciallyBspeakBsoundsBsongBsmthBslowlyBshesBshallBseenBsearchBschBrepliedBrealBreadingBreB	questionsBputBpoundsBpocketbabecoukBplayBpgBpacksBozBotherBorangeBoperatorBonesBomwBoldBokayBntBmomsBmobBminB
membershipBmeanBmalariaBmailBmadeBlucozadecoukwrcBlucozadeBltdBlovingBlessonBleBlandBjuzBjordanBjokeBitselfBitcouldB
interestedB	insuranceB	inclusiveB	importantBhurtBhurryBhungryBhoursBhotBholidayBholdBhmmBhitBheeBhavingBgymBgoodmorningBgirlsBgirlBfriendBfreemsgB
footprintsBfoneBfeelsBfarBexamBetcBenterBenjoyBenergyBdriveBdrawBdntBdetailsBdateBdarlinBcryBcreditsBcraveBcrabBcountryBcomputerBcolourBcollegeBcoinsBcharityBchargeBcellBcarlosB	camcorderBcallsBbusBbunsBbtBboytoyBbothBbookedBbonusBblooBbitBbirthdayBbigBbadBbabyBawayBauctionBaskedBarrangeBanywayBanytimeBanswerBalrightBaddressBactuallyB86688B8007B750B7B61200B150ppmB150pB12B10pminB0800Bå£800Bå£400Bå£200Bå£1250ByuouByouveByoullByouhowByepBxxxBxavierBwunBwrongBwriteBwotB	wonderfulBwklyBwifBwholeBwheresBwhateverBweeklyBweedBwedBwavesBwaterBwapBwantsBvlBvillageBviaBvalidB
valentinesBuuBusualBuserBurnBupdateB	unlimitedBunlessBullBukBughBttylBtruthBtrouserBtowardsBtouchBtotBtookBtoniteBtisBtiredBtimingBtimesBticketsBthrowBthinkinB	thatåÕsBtenBtcsbcm4235wc1n3xxBtcBtariffsBtamilnaduthenBtakingBswingBsurpriseBsurfingBsunshineBsugarBstudyingBstudyBstressBstoryBstitchBsthBstatusBstampsBspotBspokeBspentBspendBspecsBsoundBsmokinBsmileBslowBsleepingandBsleepBslaveBskypeBsk38xhBsisterBsilentBshowingBshowerBshortBshopBshareBsetB	seriouslyBseriousBsecretBsecondBseasonB	searchingBseaBscroungeBsayingBsaoBsantaBsaeBsadBsacBrunningBruleBruBroww1jhlBroseBrewardBrevealBrespectBrepresentativeBreplyingBrentalBremoveB
registeredBregardsBreachBpx3748BpubBpromisesBpromiseBprojectBprofBprobablyBprabhaBpossibleBpoppedBpolysBpoboxBpoBplayerBpicturesBphotosBphonesB	performedBpayBpassportBparisBpaBoutforBoriginalBorderBopinionBoopsBonlineBoftenBofficeBoBnycBnormallyBnobodyBni8BnevaB	netcollexBnetBneedsBnearBnationalBnaBmyselfB	myparentsBmustBmuBmrBmovingBmoveBmomentsBmodelB
mobileupd8BmobilesvaryBmobilesBmmBmissingBmissinBmisBmightBmerryBmemberBmeetinBmedicalBmealBmaxå£7BmanyBmandaBmallBmahBltdecimalgtBlotBlostBlooksBlooBlongerBlogBloanBlinesBliaoBlemmeBlehBlectBleavingBlaughBlampBktvBkoteesBknowsBkkwhereBkkthisBkettodaBkatexxxBjstBjayBipodBip4BinvitedBintoBinterestBinsteadBinfowww100percentrealcomBimmaBimagineBidkBideaBhunBhuhBhttpwwwetlpcoukexpressofferBhotelBhollaBholderBhintBhgsuite3422landsBhellohowBhelloedBheavyBheartBheadBhavBhateBhardBhappensBhappenBhalf8thBhairB	hahahauseBgr8BgottoBgottenBgottaBgoinBgodBgiftBgenuineB	gentlemanBgayBgasBfutureBfuckinBfrndsBfridayBfrenBfrBforeignBflightBflatBfinishedBfigureBfeltBfaultBfastBfarmB	fantasticBfancyBfamamusBfactBfacebookBfaceBevenBerBentryBenoughBeatBeachBdryBdrinkBdreamsBdormBdonåÕtBdonB	doingwhatBdoggingBdoesntBdisturbBdisBdirectBdignityBdigitalB	differentBdiffBdenB
definitelyBdeepBdecimalBdeadBdatingBdangBdailyBdadBcutBcupBcumBcozBcountBcontractBcontentsBconfirmBcomuk220cm2B
completelyBcompleteBcompanyBcomingBcominB
collectionBcollectBcoatBcloseBclearingBclearedBcleanB	christmasB	chocolateBchildrenBchildishBcheapB	challengeBcelebBcauseBcarolineBcantdoBcampusBcallcostBbuyingBbusyBbuffBbrotherBbreakBbrainBboyBbowlBbornBblurB	bluetoothBbiggerBbetBbelieveBbehindBbedBbcumsBbcozBbathBawesomeBawardedBawakeBawaitBaveB	availableBaudreysBattenBassBashleyBargumentB	aretakingBaprilBappsB
appreciateBanythingtomorrowBansBaloneBalcoholBagoBagesB	affectionBaddB9aeB88600B86021B80062B8B7250iB6031B6B5weB500B50B3qxj9B2whenB18B169B12hrsB10pB10am9pmB100B09071512433B08712460324B087123002209am7pmB08702840625B	087016248B08700621170150pB08000839402B050703B‰ÛB	å£75000Bå£5Bå£350Bå£300Bå£250B å£1minmobsmorelkpobox177hp51flBå£150weekB2å£150pmmorefrommobile2bremovedmobypobox734ls27yfB
å£100000Bå£100BzedBzahersByunnyByummyByoyyoooByowifesByoursByoumyByouldByouiByoudB	youcarlosByorgeByogaBymB	yesterdayByeshereByellowByeahandByayByavntByarByahooBxxBxchatBx2Bwwwtxt2shopcomBwwwrtfsphostingcomBwwwringtonescoukB
wwwdbuknetBwwwclubmobycomBwudBwthoutBwthBwtfBwtBwronglyBwrkinBwrkiBwrkBworthBworstBworseBworldsBworkinBwomanBwntBwlcomeBwivBwitoutBwithinBwishesBwinsBwinedBwineBwindowsBwindowBwindBwillingBwildBwhrBwhomBwhereverBwheneverBweveB	westshoreBwesleysBwelltakeBwelcomeBweirdBweightB	wednesdayBweddinBwebsiteBwebpageBwearingBwearBwc1n3xxBwatllBwatchesBwasntBwarwickBwarmBwanaB	walkaboutBwalikingBwadebridgeiBwaBw14rgBwBvotedBvomitBvoilaBvisitorsBvisitingBvijayBvideopicBvideophonesB	videochatBvibrateBversionB	verifyingBvdayBvaryBvaluingBvaluedB
valid12hrsB	valentineBusmleBuselessBusefulBusbB	urfeelingBureBuploadedBuploadBunsubBunrecognizedBunniBuniqueBundrstndBunderstandingB
understandB
underdtandBunclesBuncleB	unclaimedBuksB	ujhhhhhhhBucallBtypesBtxtingBtxtinBtwentyBtrustBtruroBtripsBtripleBtripBtransfrBtradeBtoyotaBtouchedBtortureBtonsBtonguedBtones2uBtomBtollBtoiletBtoclaimBtobedBtmwBtmobileBtlkBtiminBtimiBtilBthursdayBthursBthsBthruBthroughB	thousandsBthotBthoBthinksBtherellBtheoryBthedailydrawBtheatreBthatåÕscoolBthankuBthanksgivingBtextingBtextedBtermBteresaBtensedBtenerifeBtenantsBtempBtelphoneBtellmissBtelBteeBteaseBteamsBtddnewsletteremc1coukBtbBtaughtBtauBtattoosBtatBtantrumsBtampaBtalksBtalkinBtalentBtakecareBtahanBtaggedBtBswtBsweetsB
sweetheartBswatchBsw73ssBsuxBsutraBsurelyBsupposedBsuperB	sundayishBsunB	sumthinxxBsummonBsuitsBsuddenlyB
successfulBsubscribersB	submittedBsubBstylishB	stupiditsBstupidBstudentB
stressfullBstreetBstr8BstormingB	stop2stopBstoneBstolenB	stockportBstilBstepsBsteamBstdBstayinBstationBstartedBstapatiBstBsryB	squattingBsptyroneBspreeBsportsxBspontaneouslyBspoiledBsplwatBsplendidBsplBspkB	spjanuaryBspellingBspellBspeedBspecificBspaceBspBsoupBsosoBsorydarealyfrmBsoryB	sortedbutBsortedBsortBsophasBsoonerBsonyBsomoneB	sometimesBsometimeBsomethinBsomebodyBsolBsnehamBsnatchBsnakeBsnBsmsdBsmokingBsmeoneBsmacksBslobBsleepyBsleepwellamptakeBsleepingBslapB	slaaaaaveBsixB	situationBsiteBsipixBsipBsincoBsimulateBsimplyBsimpleBsimBsigningBsideBsickBshyBshutBshowrBshoweredBshowedBshowBshovingBshortlyBshoretheBshirtBshippedBshiningBshindigBshellBshdBshbBshantBseveralB	seventeenBsevenBsettledBsemiBsemesterBsemBselfBseemedB	secondaryBscrumptiousBscreenBscreamedBscoresBscoreBschoolsBschoolBscheduleBscaryBsbutBsaysBsaveBsaucyBsaturdayB
satsgettinB	satisfiedBsarBsandiagoBsalespeeBsaleBsalaryBsafeB	sacrificeBsackedBs8BrummerBrugbyBrubBroomsBroomateBroleBrogerBrockBroadBrippedBringtonekingBringtoneBringBrichBrhodeBrgentBreviewBrevealedBretardBrestrictBrestockBresponcewhatBresearchBreportBreplaceBrepentBrentlB	rencontreBremovalBrembBrelocateBrelaxBrefusedBrefundB	refreshedB
referencesBreeBreduceBrecordsBrechargerakheshB	receivingBreceivedBrecdBreassuranceBrealizeBrealityB	readinessBreadBreactingBratherBratesBranjithBrangeBrangBrandyBrandomlyBrandomBrajiniBraiseBraidenBquizwinBquizBquiteamuzingBquickBquestionBquesBqualityBqiBqBpushesBpushBpurseBpurposeB
punishmentBpublishBpt2BpropsBproperB	promotingBpromoBprofitBprocessBprobthatBproblemB
prizeswithBprintBprincesBprincegnBprinceB
previouslyBpressiesBpresntsBprescriptionBprepareB
preferablyBpreBprayBprakasamanuBprakasamB
practisingBprabuBprabhaimBppm150B	potentialBpostcardBpostBpositiveB	positionsBpositionBpoorlyBpoorBpoopBpoolBpolytruepixringtonesgamesB
polyphonicBpolyphBpoly200pBpolyBpoliceBpokingBpokerBpodBpocketsBpobox84Bpobox334Bpobox11414tcrw1Bpobox1BplumBplentyBpleassssssseeeeeeBplayinBplayersBplayedBplansBplanningBplannedBplacesB	placementBpixBpissedBpiscesBpigBpickedBpiahBphotoBphonedBphneBphasingBpharmacyBpeteBpersonsBpermissionsBpenisBpenBpeakBpeaceBpaypalBpayeeBpaybackBpattysBpattyB	pataisthaBpasswordBpassionBpassedBparentsiBparentsB
parantellaB	parachuteBpapersBpapBpantsBpackingBpackageBpBoverdoseBoutsideBoutsBothersBorediBoptBopenBooohBontoBonluyBoniBonedgeB
olayiwolasBoilB	ogunrindeBofficewhatsB	offcampusBoccupyB	objectionBo2BnytBnysBnwBnverBnumbersB	numberplsBnttBnowiBnotifiedBnothinBnosyBnosBnormalBnopeB	nooooooooBnolineBnokiasB	nokia150pBnoitsBnobutBnite2BnigeriaBnickBni8swtBnewsBnewquaysendBnevrBnelsonBnegativeBneekunnaBneedingBnecesityBnauseousBnatureBnanBnamesBnamedBnahBnagarBmusicBmumsBmummysBmummyBmultiplyBmtalkBmtBmsgsBmrngBmp3BmoviesBmovedB	mountainsBmotorolaBmotivateBmotherfuckerBmorphineBmorninBmoraloneBmoonsBmonthsBmonthlyBmonl8rsxB	mondaynxtBmojiBmofoBmodeBmobyBmobsicomBmnthsBmnsBmlBmiteBmisundrstudBmistakesBmistakeBmissedBmisscallB	miserableB
misbehavedBminuteBminorBmineBmilkBmileageBmilaBmetBmessagesB	messagenoBmelniteBmehBmedsBmeantBmeaninglessB
meaningfulBmeaningB	max10minsBmaturedBmatthewBmattersB	mattermsgBmatesBmateBmatchBmastersBmassagesBmarryB	marketingBmargaretBmarchBmapsBmapBmaneeshaBmandanB
managementBmaleBmakinBmakesBmajorBmagicBmagBmachoBmacBm95Bm8sBm39m51Bm26BlyricsBlunchBluckyBluck2BluckBlturlgtBlt3BloyalBlovesBloverboyBloverBlovejenBlovBlousyBloudBlottoBlossBlosesBlooseBlookingBlogoffBlodgingB	locationsBlocalBloadsBlmaoBlkBlipsBlionB
linerentalBlinedBlimpingBlimitingBlikesBliftBlifetimeBlieBlibraryBletterBlessonsBlessBleonaBleiBlegalBleftBleavesBleastBldnw15hBldnBlccltdBlazyBlargestBlaptopBlapBlaneB	landlinesBlandingBlaidBlagBladyBladiesBla3BkusruthiBknowheBkkimBkitsBkissesBkindlyBkindaBkindBkillBkickBkeyBkeepingBkeBkbutBkbBkateB
kanoanywayBkanoBkamaBkallisBkalaachutaaramaBkafterBjpB	jordantxtBjontinBjokingBjoinBjogBjideBjiaBjewelryBjesusBjeevithathileBjaysBjayaBjavaBjaklinBjadaBjackpotB
i‰Û÷llBizzitBizBitletBissuesBisntBisloveBislandBisareBirulinaeB	irritatedBiqBinvntedB
invitationBinvestigateB	interfuedBinterestingB	intentionBintelligentBinsideBinningsBinformationBinfoBindependentlyB
incrementsBinconvenientBincBimpBimmediatelyBillnessBilBifinkBidiotBidconveyBicicibankcomBicicBibhltdBiasBhydeBhwdBhustleBhurtingBhungoverB	hundredheBhundredBhudgiBhttpwwwurawinnercomBhrsBhrishiBhrB
housewivesBhotelsBhostelBhospitalBhornyBhoneyB
homeownersBhmmmmBhmmmBhmBhlBhitteranywayBhighBhideBhexBhelp08718728876Bhelp08700469649BhellaBhelensBhelenBheavenBheartsBhearinBheardB	headstartBheadsetBhavntBhauntBhaulBhasntBharlemB	happinessBhappiestBhappendBhangingBhanginBhangerBhandleBhamBhabitBguyBguiltyBguideBguaBgtBgrrBgrinderBgravityBgravelBgraveBgraphicsBgrantedBgrandmasBgrandB	grahmbellB	graduatedBgotoBgotaBgorgeousBgoogleBgoodtimeoliBgoodeveningBgoneuBgoneBgnBgloBgivitsBgivesBgirlfrndBgiftsBgfBgetsBgeorgesBgentlyBgeniusBgeeeeBgeeBgeBgdBgbBgaveBgaugeBgastroenteritisBgangBgameBgalnoBgalileoBgalcanBgalBfunnyBfunkB	functionsBfullonsmscomBfullBfulfilBfulBfuckingBfuckedBfrommBfrndzB	frndsshipBfrndshipBfrensBfreezingBfree2dayBfreakedBfraudsBfoxBfowlerBfourthBfourBfoundBforthBformatBforfeitBforeverBforceBfoodBfonesB	followingBfollowedBflipBflashBfixesBfixedBfixBfinishesBfinewhenBfinalBfilmBfillsBfiguringBfightsBfemaleBfeetBfeelingBfebruaryBfeaturesBfavourBfaveBfasterB	fantasiesBfallingBfallBfairBfaberBfBextBexplainB
experimentB
experienceB	expensiveB	expectingBexhaustBexerciseBexcuseBexcitedB	excellentBexactlyBevngB	everythinBeveryoneBeverydayB
everybodysB	everybodyBevery1BeventsBeveBespellBerythingBertiniBerrBenuffBentitledBenteredBennalBennaBenglishBenglandBenginB
embarassedBehB	efficientBedgeBechoBec2aBeatingBeatenBeasierBearnB	earlierweBearlierBdwnBdvdBduringBdumbBduchessBdrunkBdrugsBdroppedBdresserB
dreamsmuahBdozensB	downloadsBdoughBdotBdoorBdonewantBdollarsBdoesn‰Û÷tB	documentsBdoctorBdocsBdobbysBdobbyBdloadBdledBdivisionBdivertBdittoBdistanceBdiscountBdisconnectedBdiscloseBdisappearedBdinedBdineB	dimensionBdifferencesBdietingBdieBdidn‰Û÷tB
dictionaryBdickBdialBdeyBdetailsiB	desperateBdenaBdeluxeB	deliveredBdeliverBdelB	definitlyBdecideBdecemberBdecBdearslpBdearlyB
dearlovingBdeariBdealBdaysheBdawnsBdatsBdatesBdarlingBdarknessBdanallaBdaddyBdabooksBcutestB	customersBcustcareBcustBcurtseyB	currentlyBcummingBcuddlingBcudBcthenBcsBcruiseBcroydonBcreditBcreativeBcrazyBcr9BcozsomtimesBcoverBcourseBcouldntBcostå£150pmBcostumesBcostumeBcostaBcorrectBcornwallBcopsB
convincingB	convertedBcontentB	contactedBconnectionsB
connectionBconfusesB
conclusionBconcertBcompromisedBcomplimentaryB
completingB
complacentBcompBcommonBcommandBcomBcollegB
coherentlyBcoBclothesBclosingdate040902BcliffBclearlyBclassicB	claimcodeBcitylinkBchooseBchoiceBchikkugoingBchikkuBcherishBcheeseBcheckedBcheaperBchasingBcharmingB
charactersB	characterBchangedBchaBcertificateB	celebrateBceleb4BccnaBcatchesBcatchBcarsBcarryBcarlyBcareswtB	carefullyBcarefulBcaredBcardB
cappuccinoB	cancelledBcanceledBcanadaBcamryBcalmBcallinBcallerBcallbackBcall2optoutlf56BcaliBcalBcabBc52Bbx526Bbx420BbusinessBbunchBbuildingBbudB
btnationalBbruvBbruceBbrothasBbroBbritishBbringingBbringB	brilliantBbreatherBbreatheBbreakerBbrbBboyyBbox420Bbox334BboxBboutxxBboutBbostonBboringBbooBbongBboldBbognorBbodyBblueBblondeBblokesBblockBblimeyBblessingBblankB
blackberryBblackB	birthdateBbiroBbillBbiggestBbffsBbfBbeyondBbewareBbettersnBbenBbeliveBbehaveBbeerBbecomeBbecausetheyBbecausBbearsBbczBbcm4284BbbsBbbdeluxeBbbBbatteryBbatsmanBbatB	basicallyBbasicBbariBbarBbankBbamBbaigBbabyhopeBbabesBbaBawaitingBavoBaventBavailabletheyBavaBauntysBauntyBauditionB
audiitionsB	attendingBatlantaBateBaskingBarunBartsBarmandBargueBarentBaquariusBappreciatedBappointmentsBappointmentBapplyingBapply2BapplicationB
apparentlyB	apologiseBapartBanywhereBanywaysBanythinB	anythiingBanyoneBanswersBansrBanotBannoncementBannaBamyBamtBamsBalthoughBalsoorBalmostBaliveBalgarveBalfieBairportBairBaidBahsenBagentsBage23BagainstBagB	adventureBadpBadoreBadmirerBaddsBactivateBactingBacrossBachanammarakheshqatarBaccountsBaccordinglyBaccidentB
accessibleBabjBabiBaaBa30B9B8pmB89080B87575B87239B85023B84484B84122B83383B83355B
82277unsubB82277B81010B80878B80182B78pminB730B6missedB69888B6986618B69101B630B6230B5wbB5gentlyB5freeB530B4wrdB4uB4pressB4luxB4infoB4getB49557B46B449071512431B449050000301B447perB4403ldnw1a7rw18B430B42810B42478B41685B40gbB40B3uzB3minsB3gbpB3gB3dB3coverB391784B3650B3510iB
31pmsg150pB30pptxtB300603B3000B2wuB2waxstoB2u2B2stoptxtB	2nitetellB2niteB2mrwB2morroB29100B28B261104B250B241004B230ishB20pB200B1x150pwkB1imB1hrB1956669B15pminB150ptoneB150prcvdB150pmtmsgrcvd18B150B139B1327B11mthsB1172B116B11B10ppmB09099726481B09066382422B09066380611B09066368327B09066361921B09065174042B09063458130B09063440451B09061221066B09058098002B09058095201B09058094583B09058091870B09050090044B09050003091B09050001808B09050000301B08719899230B08718738001B08718730666B08717898035B08717509990B08717507711B08715705022B08715203677B08715203652B08712402779B08712402050B08712300220B0870kB08709501522B08709222922B08708034412B08707509020B0870241182716B08701417012B08452810073B08450542832B08002986906B07821230901B07753741225B071104B07046744435B02070836089B008704050406
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
H
Const_4Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
Adam/v/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_7/bias
w
'Adam/v/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_7/bias
w
'Adam/m/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/v/dense_7/kernel

)Adam/v/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/kernel*
_output_shapes

: *
dtype0
�
Adam/m/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/m/dense_7/kernel

)Adam/m/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/kernel*
_output_shapes

: *
dtype0
~
Adam/v/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/v/dense_6/bias
w
'Adam/v/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/bias*
_output_shapes
: *
dtype0
~
Adam/m/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/m/dense_6/bias
w
'Adam/m/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/v/dense_6/kernel

)Adam/v/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/kernel*
_output_shapes

:@ *
dtype0
�
Adam/m/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/m/dense_6/kernel

)Adam/m/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/kernel*
_output_shapes

:@ *
dtype0
~
Adam/v/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/v/dense_5/bias
w
'Adam/v/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/bias*
_output_shapes
:@*
dtype0
~
Adam/m/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/m/dense_5/bias
w
'Adam/m/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/v/dense_5/kernel

)Adam/v/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/kernel*
_output_shapes

:@*
dtype0
�
Adam/m/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/m/dense_5/kernel

)Adam/m/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/kernel*
_output_shapes

:@*
dtype0
�
Adam/v/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�N*,
shared_nameAdam/v/embedding/embeddings
�
/Adam/v/embedding/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding/embeddings*
_output_shapes
:	�N*
dtype0
�
Adam/m/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�N*,
shared_nameAdam/m/embedding/embeddings
�
/Adam/m/embedding/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding/embeddings*
_output_shapes
:	�N*
dtype0
�
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_27583*
value_dtype0	
m

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name32098*
value_dtype0	
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

: *
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
: *
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:@ *
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@*
dtype0
�
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�N*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	�N*
dtype0
s
serving_default_examplesPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_examples
hash_tableConst_5Const_4Const_3embedding/embeddingsdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_55188
�
StatefulPartitionedCall_1StatefulPartitionedCall
hash_tableConst_1Const*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__initializer_55686
�
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__initializer_55698
:
NoOpNoOp^PartitionedCall^StatefulPartitionedCall_1
�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
�E
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*�D
value�DB�D B�D
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
		tft_layer

signatures*
* 

	keras_api* 
;
	keras_api
_lookup_layer
_adapt_function*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias*
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias*
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
$B _saved_model_loader_tracked_dict* 
5
1
*2
+3
24
35
:6
;7*
5
0
*1
+2
23
34
:5
;6*
* 
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Htrace_0
Itrace_1* 

Jtrace_0
Ktrace_1* 
/
L	capture_1
M	capture_2
N	capture_3* 
�
O
_variables
P_iterations
Q_learning_rate
R_index_dict
S
_momentums
T_velocities
U_update_step_xla*

Vserving_default* 
* 
* 
7
W	keras_api
Xlookup_table
Ytoken_counts*

Ztrace_0* 

0*

0*
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

`trace_0* 

atrace_0* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

gtrace_0* 

htrace_0* 

*0
+1*

*0
+1*
* 
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

20
31*

20
31*
* 
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

utrace_0* 

vtrace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

:0
;1*

:0
;1*
* 
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

|trace_0* 

}trace_0* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
y
�	_imported
�_wrapped_function
�_structured_inputs
�_structured_outputs
�_output_to_inputs_map* 
* 
C
0
1
2
3
4
5
6
7
	8*

�0
�1*
* 
* 
/
L	capture_1
M	capture_2
N	capture_3* 
/
L	capture_1
M	capture_2
N	capture_3* 
/
L	capture_1
M	capture_2
N	capture_3* 
/
L	capture_1
M	capture_2
N	capture_3* 
* 
* 
* 
�
P0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
�0
�1
�2
�3
�4
�5
�6*
<
�0
�1
�2
�3
�4
�5
�6*
* 
/
L	capture_1
M	capture_2
N	capture_3* 
* 
V
�_initializer
�_create_resource
�_initialize
�_destroy_resource* 
�
�_create_resource
�_initialize
�_destroy_resourceJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*

�	capture_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�created_variables
�	resources
�trackable_objects
�initializers
�assets
�
signatures
$�_self_saveable_object_factories
�transform_fn* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
f`
VARIABLE_VALUEAdam/m/embedding/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/embedding/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_5/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_5/kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_5/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_5/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_6/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_6/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_6/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_6/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_7/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_7/kernel2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_7/bias2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_7/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 

�serving_default* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
"
�	capture_1
�	capture_2* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	iterationlearning_rateAdam/m/embedding/embeddingsAdam/v/embedding/embeddingsAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biastotal_1count_1totalcount?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1Const_6**
Tin#
!2	*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_55925
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding/embeddingsdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	iterationlearning_rateMutableHashTableAdam/m/embedding/embeddingsAdam/v/embedding/embeddingsAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biastotal_1count_1totalcount*(
Tin!
2*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_56018��	
�

�
B__inference_dense_6_layer_call_and_return_conditional_losses_55399

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
q
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_55276

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�

�
B__inference_dense_5_layer_call_and_return_conditional_losses_55635

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
D__inference_embedding_layer_call_and_return_conditional_losses_55604

inputs	)
embedding_lookup_55599:	�N
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_55599inputs*
Tindices0	*)
_class
loc:@embedding_lookup/55599*+
_output_shapes
:���������d*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������du
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������d5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:%!

_user_specified_name55599
�
�
__inference_restore_fn_55727
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity��2MutableHashTable_table_restore/LookupTableImportV2�
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: W
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1:,(
&
_user_specified_nametable_handle
��
�	
 __inference__wrapped_model_55270
text_xfa
]model_2_text_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleb
^model_2_text_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	8
4model_2_text_vectorization_1_string_lookup_1_equal_y;
7model_2_text_vectorization_1_string_lookup_1_selectv2_t	;
(model_2_embedding_embedding_lookup_55242:	�N@
.model_2_dense_5_matmul_readvariableop_resource:@=
/model_2_dense_5_biasadd_readvariableop_resource:@@
.model_2_dense_6_matmul_readvariableop_resource:@ =
/model_2_dense_6_biasadd_readvariableop_resource: @
.model_2_dense_7_matmul_readvariableop_resource: =
/model_2_dense_7_biasadd_readvariableop_resource:
identity��&model_2/dense_5/BiasAdd/ReadVariableOp�%model_2/dense_5/MatMul/ReadVariableOp�&model_2/dense_6/BiasAdd/ReadVariableOp�%model_2/dense_6/MatMul/ReadVariableOp�&model_2/dense_7/BiasAdd/ReadVariableOp�%model_2/dense_7/MatMul/ReadVariableOp�"model_2/embedding/embedding_lookup�Pmodel_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2u
"model_2/tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
model_2/tf.reshape_1/ReshapeReshapetext_xf+model_2/tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
(model_2/text_vectorization_1/StringLowerStringLower%model_2/tf.reshape_1/Reshape:output:0*#
_output_shapes
:����������
/model_2/text_vectorization_1/StaticRegexReplaceStaticRegexReplace1model_2/text_vectorization_1/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite o
.model_2/text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
6model_2/text_vectorization_1/StringSplit/StringSplitV2StringSplitV28model_2/text_vectorization_1/StaticRegexReplace:output:07model_2/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
<model_2/text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
>model_2/text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
>model_2/text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
6model_2/text_vectorization_1/StringSplit/strided_sliceStridedSlice@model_2/text_vectorization_1/StringSplit/StringSplitV2:indices:0Emodel_2/text_vectorization_1/StringSplit/strided_slice/stack:output:0Gmodel_2/text_vectorization_1/StringSplit/strided_slice/stack_1:output:0Gmodel_2/text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
>model_2/text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
@model_2/text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
@model_2/text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
8model_2/text_vectorization_1/StringSplit/strided_slice_1StridedSlice>model_2/text_vectorization_1/StringSplit/StringSplitV2:shape:0Gmodel_2/text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Imodel_2/text_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Imodel_2/text_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
_model_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast?model_2/text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
amodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastAmodel_2/text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
imodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapecmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
imodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
hmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdrmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0rmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
mmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterqmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0vmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
hmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastomodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
gmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxcmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
imodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
gmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2pmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0rmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
gmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMullmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumemodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumemodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0omodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
qmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapecmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0zmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
lmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincounttmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0omodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0tmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
fmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
amodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumsmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0omodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
jmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
fmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
amodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2smodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0gmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0omodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Pmodel_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2]model_2_text_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handle?model_2/text_vectorization_1/StringSplit/StringSplitV2:values:0^model_2_text_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
2model_2/text_vectorization_1/string_lookup_1/EqualEqual?model_2/text_vectorization_1/StringSplit/StringSplitV2:values:04model_2_text_vectorization_1_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
5model_2/text_vectorization_1/string_lookup_1/SelectV2SelectV26model_2/text_vectorization_1/string_lookup_1/Equal:z:07model_2_text_vectorization_1_string_lookup_1_selectv2_tYmodel_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
5model_2/text_vectorization_1/string_lookup_1/IdentityIdentity>model_2/text_vectorization_1/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:���������{
9model_2/text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
1model_2/text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������d       �
@model_2/text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor:model_2/text_vectorization_1/RaggedToTensor/Const:output:0>model_2/text_vectorization_1/string_lookup_1/Identity:output:0Bmodel_2/text_vectorization_1/RaggedToTensor/default_value:output:0Amodel_2/text_vectorization_1/StringSplit/strided_slice_1:output:0?model_2/text_vectorization_1/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
"model_2/embedding/embedding_lookupResourceGather(model_2_embedding_embedding_lookup_55242Imodel_2/text_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*;
_class1
/-loc:@model_2/embedding/embedding_lookup/55242*+
_output_shapes
:���������d*
dtype0�
+model_2/embedding/embedding_lookup/IdentityIdentity+model_2/embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:���������d{
9model_2/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'model_2/global_average_pooling1d_1/MeanMean4model_2/embedding/embedding_lookup/Identity:output:0Bmodel_2/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
%model_2/dense_5/MatMul/ReadVariableOpReadVariableOp.model_2_dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model_2/dense_5/MatMulMatMul0model_2/global_average_pooling1d_1/Mean:output:0-model_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&model_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_2/dense_5/BiasAddBiasAdd model_2/dense_5/MatMul:product:0.model_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
model_2/dense_5/ReluRelu model_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
%model_2/dense_6/MatMul/ReadVariableOpReadVariableOp.model_2_dense_6_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
model_2/dense_6/MatMulMatMul"model_2/dense_5/Relu:activations:0-model_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&model_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_2/dense_6/BiasAddBiasAdd model_2/dense_6/MatMul:product:0.model_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� p
model_2/dense_6/ReluRelu model_2/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
%model_2/dense_7/MatMul/ReadVariableOpReadVariableOp.model_2_dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
model_2/dense_7/MatMulMatMul"model_2/dense_6/Relu:activations:0-model_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&model_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_2/dense_7/BiasAddBiasAdd model_2/dense_7/MatMul:product:0.model_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
model_2/dense_7/SigmoidSigmoid model_2/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentitymodel_2/dense_7/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^model_2/dense_5/BiasAdd/ReadVariableOp&^model_2/dense_5/MatMul/ReadVariableOp'^model_2/dense_6/BiasAdd/ReadVariableOp&^model_2/dense_6/MatMul/ReadVariableOp'^model_2/dense_7/BiasAdd/ReadVariableOp&^model_2/dense_7/MatMul/ReadVariableOp#^model_2/embedding/embedding_lookupQ^model_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : : : 2P
&model_2/dense_5/BiasAdd/ReadVariableOp&model_2/dense_5/BiasAdd/ReadVariableOp2N
%model_2/dense_5/MatMul/ReadVariableOp%model_2/dense_5/MatMul/ReadVariableOp2P
&model_2/dense_6/BiasAdd/ReadVariableOp&model_2/dense_6/BiasAdd/ReadVariableOp2N
%model_2/dense_6/MatMul/ReadVariableOp%model_2/dense_6/MatMul/ReadVariableOp2P
&model_2/dense_7/BiasAdd/ReadVariableOp&model_2/dense_7/BiasAdd/ReadVariableOp2N
%model_2/dense_7/MatMul/ReadVariableOp%model_2/dense_7/MatMul/ReadVariableOp2H
"model_2/embedding/embedding_lookup"model_2/embedding/embedding_lookup2�
Pmodel_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2Pmodel_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:���������
!
_user_specified_name	text_xf:,(
&
_user_specified_nametable_handle:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_user_specified_name55242:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
'__inference_dense_6_layer_call_fn_55644

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_55399o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:%!

_user_specified_name55638:%!

_user_specified_name55640
�
F
__inference__creator_55694
identity: ��MutableHashTable�
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_27583*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: 5
NoOpNoOp^MutableHashTable*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
�
.
__inference__initializer_55698
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�F
�
__inference_adapt_step_54449
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	��IteratorGetNext�(None_lookup_table_find/LookupTableFindV2�,None_lookup_table_insert/LookupTableInsertV2�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:���������*"
output_shapes
:���������*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:����������
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountWStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:���������:���������:���������*
out_idx0	�
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:�
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:,(
&
_user_specified_nametable_handle:

_output_shapes
: 
�
�
'__inference_dense_5_layer_call_fn_55624

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_55383o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:%!

_user_specified_name55618:%!

_user_specified_name55620
�
R
__inference_pruned_55027

inputs	
inputs_1
identity	

identity_1Q
inputs_copyIdentityinputs*
T0	*'
_output_shapes
:���������\
IdentityIdentityinputs_copy:output:0*
T0	*'
_output_shapes
:���������U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:���������[
StringLowerStringLowerinputs_1_copy:output:0*'
_output_shapes
:���������^

Identity_1IdentityStringLower:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:- )
'
_output_shapes
:���������:-)
'
_output_shapes
:���������
�
,
__inference__destroyer_55690
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
m
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_55302
text
identityG
ShapeShapetext*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
Shape_1Shapetext*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:����������
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������
PartitionedCallPartitionedCallPlaceholderWithDefault:output:0text*
Tin
2	*
Tout
2	*:
_output_shapes(
&:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *!
fR
__inference_pruned_55027`
IdentityIdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:M I
'
_output_shapes
:���������

_user_specified_nametext
�
,
__inference__destroyer_55702
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
'__inference_model_2_layer_call_fn_55550
text_xf
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	�N
	unknown_4:@
	unknown_5:@
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_55496o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	text_xf:%!

_user_specified_name55526:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_user_specified_name55534:%!

_user_specified_name55536:%!

_user_specified_name55538:%!

_user_specified_name55540:%	!

_user_specified_name55542:%
!

_user_specified_name55544:%!

_user_specified_name55546
�
�
D__inference_embedding_layer_call_and_return_conditional_losses_55368

inputs	)
embedding_lookup_55363:	�N
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_55363inputs*
Tindices0	*)
_class
loc:@embedding_lookup/55363*+
_output_shapes
:���������d*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:���������du
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:���������d5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:%!

_user_specified_name55363
�
V
:__inference_global_average_pooling1d_1_layer_call_fn_55609

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_55276i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_55188
examples
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	�N
	unknown_4:@
	unknown_5:@
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_serve_tf_examples_fn_55160o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:���������
"
_user_specified_name
examples:%!

_user_specified_name55164:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_user_specified_name55172:%!

_user_specified_name55174:%!

_user_specified_name55176:%!

_user_specified_name55178:%	!

_user_specified_name55180:%
!

_user_specified_name55182:%!

_user_specified_name55184
�
~
)__inference_embedding_layer_call_fn_55596

inputs	
unknown:	�N
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_55368s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:%!

_user_specified_name55592
�

�
B__inference_dense_7_layer_call_and_return_conditional_losses_55675

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
__inference__initializer_556868
4key_value_init32097_lookuptableimportv2_table_handle0
,key_value_init32097_lookuptableimportv2_keys2
.key_value_init32097_lookuptableimportv2_values	
identity��'key_value_init32097/LookupTableImportV2�
'key_value_init32097/LookupTableImportV2LookupTableImportV24key_value_init32097_lookuptableimportv2_table_handle,key_value_init32097_lookuptableimportv2_keys.key_value_init32097_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: L
NoOpNoOp(^key_value_init32097/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :�:�2R
'key_value_init32097/LookupTableImportV2'key_value_init32097/LookupTableImportV2:, (
&
_user_specified_nametable_handle:A=

_output_shapes	
:�

_user_specified_namekeys:C?

_output_shapes	
:�
 
_user_specified_namevalues
�
q
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_55615

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
��
�
__inference__traced_save_55925
file_prefix>
+read_disablecopyonread_embedding_embeddings:	�N9
'read_1_disablecopyonread_dense_5_kernel:@3
%read_2_disablecopyonread_dense_5_bias:@9
'read_3_disablecopyonread_dense_6_kernel:@ 3
%read_4_disablecopyonread_dense_6_bias: 9
'read_5_disablecopyonread_dense_7_kernel: 3
%read_6_disablecopyonread_dense_7_bias:,
"read_7_disablecopyonread_iteration:	 0
&read_8_disablecopyonread_learning_rate: G
4read_9_disablecopyonread_adam_m_embedding_embeddings:	�NH
5read_10_disablecopyonread_adam_v_embedding_embeddings:	�NA
/read_11_disablecopyonread_adam_m_dense_5_kernel:@A
/read_12_disablecopyonread_adam_v_dense_5_kernel:@;
-read_13_disablecopyonread_adam_m_dense_5_bias:@;
-read_14_disablecopyonread_adam_v_dense_5_bias:@A
/read_15_disablecopyonread_adam_m_dense_6_kernel:@ A
/read_16_disablecopyonread_adam_v_dense_6_kernel:@ ;
-read_17_disablecopyonread_adam_m_dense_6_bias: ;
-read_18_disablecopyonread_adam_v_dense_6_bias: A
/read_19_disablecopyonread_adam_m_dense_7_kernel: A
/read_20_disablecopyonread_adam_v_dense_7_kernel: ;
-read_21_disablecopyonread_adam_m_dense_7_bias:;
-read_22_disablecopyonread_adam_v_dense_7_bias:+
!read_23_disablecopyonread_total_1: +
!read_24_disablecopyonread_count_1: )
read_25_disablecopyonread_total: )
read_26_disablecopyonread_count: J
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	
savev2_const_6
identity_55��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: }
Read/DisableCopyOnReadDisableCopyOnRead+read_disablecopyonread_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp+read_disablecopyonread_embedding_embeddings^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�N*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Nb

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�N{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_dense_5_kernel^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0m

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@c

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:@y
Read_2/DisableCopyOnReadDisableCopyOnRead%read_2_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp%read_2_disablecopyonread_dense_5_bias^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_dense_6_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:@ y
Read_4/DisableCopyOnReadDisableCopyOnRead%read_4_disablecopyonread_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp%read_4_disablecopyonread_dense_6_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_dense_7_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

: y
Read_6/DisableCopyOnReadDisableCopyOnRead%read_6_disablecopyonread_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp%read_6_disablecopyonread_dense_7_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_7/DisableCopyOnReadDisableCopyOnRead"read_7_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp"read_7_disablecopyonread_iteration^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_learning_rate^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_adam_m_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_adam_m_embedding_embeddings^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�N*
dtype0o
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Nf
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	�N�
Read_10/DisableCopyOnReadDisableCopyOnRead5read_10_disablecopyonread_adam_v_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp5read_10_disablecopyonread_adam_v_embedding_embeddings^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�N*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Nf
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	�N�
Read_11/DisableCopyOnReadDisableCopyOnRead/read_11_disablecopyonread_adam_m_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp/read_11_disablecopyonread_adam_m_dense_5_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_12/DisableCopyOnReadDisableCopyOnRead/read_12_disablecopyonread_adam_v_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp/read_12_disablecopyonread_adam_v_dense_5_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_13/DisableCopyOnReadDisableCopyOnRead-read_13_disablecopyonread_adam_m_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp-read_13_disablecopyonread_adam_m_dense_5_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_14/DisableCopyOnReadDisableCopyOnRead-read_14_disablecopyonread_adam_v_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp-read_14_disablecopyonread_adam_v_dense_5_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_15/DisableCopyOnReadDisableCopyOnRead/read_15_disablecopyonread_adam_m_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp/read_15_disablecopyonread_adam_m_dense_6_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_16/DisableCopyOnReadDisableCopyOnRead/read_16_disablecopyonread_adam_v_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp/read_16_disablecopyonread_adam_v_dense_6_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_17/DisableCopyOnReadDisableCopyOnRead-read_17_disablecopyonread_adam_m_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp-read_17_disablecopyonread_adam_m_dense_6_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnRead-read_18_disablecopyonread_adam_v_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp-read_18_disablecopyonread_adam_v_dense_6_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_19/DisableCopyOnReadDisableCopyOnRead/read_19_disablecopyonread_adam_m_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp/read_19_disablecopyonread_adam_m_dense_7_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_20/DisableCopyOnReadDisableCopyOnRead/read_20_disablecopyonread_adam_v_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp/read_20_disablecopyonread_adam_v_dense_7_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_21/DisableCopyOnReadDisableCopyOnRead-read_21_disablecopyonread_adam_m_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp-read_21_disablecopyonread_adam_m_dense_7_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead-read_22_disablecopyonread_adam_v_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp-read_22_disablecopyonread_adam_v_dense_7_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_23/DisableCopyOnReadDisableCopyOnRead!read_23_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp!read_23_disablecopyonread_total_1^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_24/DisableCopyOnReadDisableCopyOnRead!read_24_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp!read_24_disablecopyonread_count_1^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_25/DisableCopyOnReadDisableCopyOnReadread_25_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOpread_25_disablecopyonread_total^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_26/DisableCopyOnReadDisableCopyOnReadread_26_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOpread_26_disablecopyonread_count^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0savev2_const_6"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *,
dtypes"
 2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_54Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_55IdentityIdentity_54:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_55Identity_55:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : ::: 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:40
.
_user_specified_nameembedding/embeddings:.*
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_5/bias:.*
(
_user_specified_namedense_6/kernel:,(
&
_user_specified_namedense_6/bias:.*
(
_user_specified_namedense_7/kernel:,(
&
_user_specified_namedense_7/bias:)%
#
_user_specified_name	iteration:-	)
'
_user_specified_namelearning_rate:;
7
5
_user_specified_nameAdam/m/embedding/embeddings:;7
5
_user_specified_nameAdam/v/embedding/embeddings:51
/
_user_specified_nameAdam/m/dense_5/kernel:51
/
_user_specified_nameAdam/v/dense_5/kernel:3/
-
_user_specified_nameAdam/m/dense_5/bias:3/
-
_user_specified_nameAdam/v/dense_5/bias:51
/
_user_specified_nameAdam/m/dense_6/kernel:51
/
_user_specified_nameAdam/v/dense_6/kernel:3/
-
_user_specified_nameAdam/m/dense_6/bias:3/
-
_user_specified_nameAdam/v/dense_6/bias:51
/
_user_specified_nameAdam/m/dense_7/kernel:51
/
_user_specified_nameAdam/v/dense_7/kernel:3/
-
_user_specified_nameAdam/m/dense_7/bias:3/
-
_user_specified_nameAdam/v/dense_7/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount:yu

_output_shapes
:
Y
_user_specified_nameA?MutableHashTable_lookup_table_export_values/LookupTableExportV2:yu

_output_shapes
:
Y
_user_specified_nameA?MutableHashTable_lookup_table_export_values/LookupTableExportV2:?;

_output_shapes
: 
!
_user_specified_name	Const_6
�v
�
B__inference_model_2_layer_call_and_return_conditional_losses_55496
text_xfY
Utext_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_1_string_lookup_1_equal_y3
/text_vectorization_1_string_lookup_1_selectv2_t	"
embedding_55476:	�N
dense_5_55480:@
dense_5_55482:@
dense_6_55485:@ 
dense_6_55487: 
dense_7_55490: 
dense_7_55492:
identity��dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�Htext_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������{
tf.reshape_1/ReshapeReshapetext_xf#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:���������s
 text_vectorization_1/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:����������
'text_vectorization_1/StaticRegexReplaceStaticRegexReplace)text_vectorization_1/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
.text_vectorization_1/StringSplit/StringSplitV2StringSplitV20text_vectorization_1/StaticRegexReplace:output:0/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
4text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
6text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
6text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
.text_vectorization_1/StringSplit/strided_sliceStridedSlice8text_vectorization_1/StringSplit/StringSplitV2:indices:0=text_vectorization_1/StringSplit/strided_slice/stack:output:0?text_vectorization_1/StringSplit/strided_slice/stack_1:output:0?text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
6text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
8text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
8text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0text_vectorization_1/StringSplit/strided_slice_1StridedSlice6text_vectorization_1/StringSplit/StringSplitV2:shape:0?text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
itext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Htext_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_1/StringSplit/StringSplitV2:values:0Vtext_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
*text_vectorization_1/string_lookup_1/EqualEqual7text_vectorization_1/StringSplit/StringSplitV2:values:0,text_vectorization_1_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
-text_vectorization_1/string_lookup_1/SelectV2SelectV2.text_vectorization_1/string_lookup_1/Equal:z:0/text_vectorization_1_string_lookup_1_selectv2_tQtext_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
-text_vectorization_1/string_lookup_1/IdentityIdentity6text_vectorization_1/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:���������s
1text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
)text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������d       �
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_1/RaggedToTensor/Const:output:06text_vectorization_1/string_lookup_1/Identity:output:0:text_vectorization_1/RaggedToTensor/default_value:output:09text_vectorization_1/StringSplit/strided_slice_1:output:07text_vectorization_1/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0embedding_55476*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_55368�
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_55276�
dense_5/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_5_55480dense_5_55482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_55383�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_55485dense_6_55487*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_55399�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_55490dense_7_55492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_55415w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^embedding/StatefulPartitionedCallI^text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2�
Htext_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2Htext_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:���������
!
_user_specified_name	text_xf:,(
&
_user_specified_nametable_handle:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_user_specified_name55476:%!

_user_specified_name55480:%!

_user_specified_name55482:%!

_user_specified_name55485:%	!

_user_specified_name55487:%
!

_user_specified_name55490:%!

_user_specified_name55492
�
�
'__inference_model_2_layer_call_fn_55523
text_xf
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	�N
	unknown_4:@
	unknown_5:@
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_55422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	text_xf:%!

_user_specified_name55499:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_user_specified_name55507:%!

_user_specified_name55509:%!

_user_specified_name55511:%!

_user_specified_name55513:%	!

_user_specified_name55515:%
!

_user_specified_name55517:%!

_user_specified_name55519
�
�
!__inference__traced_restore_56018
file_prefix8
%assignvariableop_embedding_embeddings:	�N3
!assignvariableop_1_dense_5_kernel:@-
assignvariableop_2_dense_5_bias:@3
!assignvariableop_3_dense_6_kernel:@ -
assignvariableop_4_dense_6_bias: 3
!assignvariableop_5_dense_7_kernel: -
assignvariableop_6_dense_7_bias:&
assignvariableop_7_iteration:	 *
 assignvariableop_8_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: A
.assignvariableop_9_adam_m_embedding_embeddings:	�NB
/assignvariableop_10_adam_v_embedding_embeddings:	�N;
)assignvariableop_11_adam_m_dense_5_kernel:@;
)assignvariableop_12_adam_v_dense_5_kernel:@5
'assignvariableop_13_adam_m_dense_5_bias:@5
'assignvariableop_14_adam_v_dense_5_bias:@;
)assignvariableop_15_adam_m_dense_6_kernel:@ ;
)assignvariableop_16_adam_v_dense_6_kernel:@ 5
'assignvariableop_17_adam_m_dense_6_bias: 5
'assignvariableop_18_adam_v_dense_6_bias: ;
)assignvariableop_19_adam_m_dense_7_kernel: ;
)assignvariableop_20_adam_v_dense_7_kernel: 5
'assignvariableop_21_adam_m_dense_7_bias:5
'assignvariableop_22_adam_v_dense_7_bias:%
assignvariableop_23_total_1: %
assignvariableop_24_count_1: #
assignvariableop_25_total: #
assignvariableop_26_count: 
identity_28��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�2MutableHashTable_table_restore/LookupTableImportV2�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_5_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_5_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_6_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_6_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_7_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_7_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_iterationIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_learning_rateIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:9RestoreV2:tensors:10*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*&
 _has_manual_control_dependencies(*
_output_shapes
 ^

Identity_9IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_adam_m_embedding_embeddingsIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_adam_v_embedding_embeddingsIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_m_dense_5_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_v_dense_5_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp'assignvariableop_13_adam_m_dense_5_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_v_dense_5_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_m_dense_6_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_v_dense_6_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_m_dense_6_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_v_dense_6_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_m_dense_7_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_v_dense_7_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_m_dense_7_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_v_dense_7_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*
_output_shapes
 "#
identity_28Identity_28:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:40
.
_user_specified_nameembedding/embeddings:.*
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_5/bias:.*
(
_user_specified_namedense_6/kernel:,(
&
_user_specified_namedense_6/bias:.*
(
_user_specified_namedense_7/kernel:,(
&
_user_specified_namedense_7/bias:)%
#
_user_specified_name	iteration:-	)
'
_user_specified_namelearning_rate:U
Q
#
_class
loc:@MutableHashTable
*
_user_specified_nameMutableHashTable:;7
5
_user_specified_nameAdam/m/embedding/embeddings:;7
5
_user_specified_nameAdam/v/embedding/embeddings:51
/
_user_specified_nameAdam/m/dense_5/kernel:51
/
_user_specified_nameAdam/v/dense_5/kernel:3/
-
_user_specified_nameAdam/m/dense_5/bias:3/
-
_user_specified_nameAdam/v/dense_5/bias:51
/
_user_specified_nameAdam/m/dense_6/kernel:51
/
_user_specified_nameAdam/v/dense_6/kernel:3/
-
_user_specified_nameAdam/m/dense_6/bias:3/
-
_user_specified_nameAdam/v/dense_6/bias:51
/
_user_specified_nameAdam/m/dense_7/kernel:51
/
_user_specified_nameAdam/v/dense_7/kernel:3/
-
_user_specified_nameAdam/m/dense_7/bias:3/
-
_user_specified_nameAdam/v/dense_7/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount
�

�
B__inference_dense_6_layer_call_and_return_conditional_losses_55655

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�v
�
B__inference_model_2_layer_call_and_return_conditional_losses_55422
text_xfY
Utext_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleZ
Vtext_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	0
,text_vectorization_1_string_lookup_1_equal_y3
/text_vectorization_1_string_lookup_1_selectv2_t	"
embedding_55369:	�N
dense_5_55384:@
dense_5_55386:@
dense_6_55400:@ 
dense_6_55402: 
dense_7_55416: 
dense_7_55418:
identity��dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�Htext_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������{
tf.reshape_1/ReshapeReshapetext_xf#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:���������s
 text_vectorization_1/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:����������
'text_vectorization_1/StaticRegexReplaceStaticRegexReplace)text_vectorization_1/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
.text_vectorization_1/StringSplit/StringSplitV2StringSplitV20text_vectorization_1/StaticRegexReplace:output:0/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
4text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
6text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
6text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
.text_vectorization_1/StringSplit/strided_sliceStridedSlice8text_vectorization_1/StringSplit/StringSplitV2:indices:0=text_vectorization_1/StringSplit/strided_slice/stack:output:0?text_vectorization_1/StringSplit/strided_slice/stack_1:output:0?text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
6text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
8text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
8text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
0text_vectorization_1/StringSplit/strided_slice_1StridedSlice6text_vectorization_1/StringSplit/StringSplitV2:shape:0?text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Wtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
etext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
`text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
atext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
itext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
ctext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
dtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
btext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
^text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Ytext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Htext_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Utext_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handle7text_vectorization_1/StringSplit/StringSplitV2:values:0Vtext_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
*text_vectorization_1/string_lookup_1/EqualEqual7text_vectorization_1/StringSplit/StringSplitV2:values:0,text_vectorization_1_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
-text_vectorization_1/string_lookup_1/SelectV2SelectV2.text_vectorization_1/string_lookup_1/Equal:z:0/text_vectorization_1_string_lookup_1_selectv2_tQtext_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
-text_vectorization_1/string_lookup_1/IdentityIdentity6text_vectorization_1/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:���������s
1text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
)text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������d       �
8text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_1/RaggedToTensor/Const:output:06text_vectorization_1/string_lookup_1/Identity:output:0:text_vectorization_1/RaggedToTensor/default_value:output:09text_vectorization_1/StringSplit/strided_slice_1:output:07text_vectorization_1/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0embedding_55369*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_55368�
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_55276�
dense_5/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_5_55384dense_5_55386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_55383�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_55400dense_6_55402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_55399�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_55416dense_7_55418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_55415w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^embedding/StatefulPartitionedCallI^text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):���������: : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2�
Htext_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2Htext_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:���������
!
_user_specified_name	text_xf:,(
&
_user_specified_nametable_handle:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_user_specified_name55369:%!

_user_specified_name55384:%!

_user_specified_name55386:%!

_user_specified_name55400:%	!

_user_specified_name55402:%
!

_user_specified_name55416:%!

_user_specified_name55418
�
:
__inference__creator_55679
identity��
hash_tablem

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name32098*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
�
�
'__inference_dense_7_layer_call_fn_55664

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_55415o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:%!

_user_specified_name55658:%!

_user_specified_name55660
�

�
B__inference_dense_5_layer_call_and_return_conditional_losses_55383

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�	
&__inference_serve_tf_examples_fn_55160
examplesa
]model_2_text_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handleb
^model_2_text_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value	8
4model_2_text_vectorization_1_string_lookup_1_equal_y;
7model_2_text_vectorization_1_string_lookup_1_selectv2_t	;
(model_2_embedding_embedding_lookup_55132:	�N@
.model_2_dense_5_matmul_readvariableop_resource:@=
/model_2_dense_5_biasadd_readvariableop_resource:@@
.model_2_dense_6_matmul_readvariableop_resource:@ =
/model_2_dense_6_biasadd_readvariableop_resource: @
.model_2_dense_7_matmul_readvariableop_resource: =
/model_2_dense_7_biasadd_readvariableop_resource:
identity��&model_2/dense_5/BiasAdd/ReadVariableOp�%model_2/dense_5/MatMul/ReadVariableOp�&model_2/dense_6/BiasAdd/ReadVariableOp�%model_2/dense_6/MatMul/ReadVariableOp�&model_2/dense_7/BiasAdd/ReadVariableOp�%model_2/dense_7/MatMul/ReadVariableOp�"model_2/embedding/embedding_lookup�Pmodel_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB s
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*
valueBBtextj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB �
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0*
Tdense
2*'
_output_shapes
:���������*
dense_shapes
:*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 �
transform_features_layer/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
::��v
,transform_features_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.transform_features_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transform_features_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&transform_features_layer/strided_sliceStridedSlice'transform_features_layer/Shape:output:05transform_features_layer/strided_slice/stack:output:07transform_features_layer/strided_slice/stack_1:output:07transform_features_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 transform_features_layer/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
::��x
.transform_features_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(transform_features_layer/strided_slice_1StridedSlice)transform_features_layer/Shape_1:output:07transform_features_layer/strided_slice_1/stack:output:09transform_features_layer/strided_slice_1/stack_1:output:09transform_features_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'transform_features_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
%transform_features_layer/zeros/packedPack1transform_features_layer/strided_slice_1:output:00transform_features_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
$transform_features_layer/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R �
transform_features_layer/zerosFill.transform_features_layer/zeros/packed:output:0-transform_features_layer/zeros/Const:output:0*
T0	*'
_output_shapes
:����������
/transform_features_layer/PlaceholderWithDefaultPlaceholderWithDefault'transform_features_layer/zeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������
(transform_features_layer/PartitionedCallPartitionedCall8transform_features_layer/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:0*
Tin
2	*
Tout
2	*:
_output_shapes(
&:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *!
fR
__inference_pruned_55027u
"model_2/tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
model_2/tf.reshape_1/ReshapeReshape1transform_features_layer/PartitionedCall:output:1+model_2/tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
(model_2/text_vectorization_1/StringLowerStringLower%model_2/tf.reshape_1/Reshape:output:0*#
_output_shapes
:����������
/model_2/text_vectorization_1/StaticRegexReplaceStaticRegexReplace1model_2/text_vectorization_1/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite o
.model_2/text_vectorization_1/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
6model_2/text_vectorization_1/StringSplit/StringSplitV2StringSplitV28model_2/text_vectorization_1/StaticRegexReplace:output:07model_2/text_vectorization_1/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
<model_2/text_vectorization_1/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
>model_2/text_vectorization_1/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
>model_2/text_vectorization_1/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
6model_2/text_vectorization_1/StringSplit/strided_sliceStridedSlice@model_2/text_vectorization_1/StringSplit/StringSplitV2:indices:0Emodel_2/text_vectorization_1/StringSplit/strided_slice/stack:output:0Gmodel_2/text_vectorization_1/StringSplit/strided_slice/stack_1:output:0Gmodel_2/text_vectorization_1/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
>model_2/text_vectorization_1/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
@model_2/text_vectorization_1/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
@model_2/text_vectorization_1/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
8model_2/text_vectorization_1/StringSplit/strided_slice_1StridedSlice>model_2/text_vectorization_1/StringSplit/StringSplitV2:shape:0Gmodel_2/text_vectorization_1/StringSplit/strided_slice_1/stack:output:0Imodel_2/text_vectorization_1/StringSplit/strided_slice_1/stack_1:output:0Imodel_2/text_vectorization_1/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
_model_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast?model_2/text_vectorization_1/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
amodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastAmodel_2/text_vectorization_1/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
imodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapecmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
imodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
hmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdrmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0rmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
mmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterqmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0vmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
hmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastomodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
gmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxcmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
imodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
gmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2pmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0rmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
gmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMullmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumemodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumemodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0omodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
qmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
kmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapecmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0zmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
lmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincounttmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0omodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0tmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
fmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
amodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumsmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0omodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
jmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
fmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
amodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2smodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0gmodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0omodel_2/text_vectorization_1/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Pmodel_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2LookupTableFindV2]model_2_text_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_table_handle?model_2/text_vectorization_1/StringSplit/StringSplitV2:values:0^model_2_text_vectorization_1_string_lookup_1_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
2model_2/text_vectorization_1/string_lookup_1/EqualEqual?model_2/text_vectorization_1/StringSplit/StringSplitV2:values:04model_2_text_vectorization_1_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
5model_2/text_vectorization_1/string_lookup_1/SelectV2SelectV26model_2/text_vectorization_1/string_lookup_1/Equal:z:07model_2_text_vectorization_1_string_lookup_1_selectv2_tYmodel_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
5model_2/text_vectorization_1/string_lookup_1/IdentityIdentity>model_2/text_vectorization_1/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:���������{
9model_2/text_vectorization_1/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
1model_2/text_vectorization_1/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������d       �
@model_2/text_vectorization_1/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor:model_2/text_vectorization_1/RaggedToTensor/Const:output:0>model_2/text_vectorization_1/string_lookup_1/Identity:output:0Bmodel_2/text_vectorization_1/RaggedToTensor/default_value:output:0Amodel_2/text_vectorization_1/StringSplit/strided_slice_1:output:0?model_2/text_vectorization_1/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
"model_2/embedding/embedding_lookupResourceGather(model_2_embedding_embedding_lookup_55132Imodel_2/text_vectorization_1/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*;
_class1
/-loc:@model_2/embedding/embedding_lookup/55132*+
_output_shapes
:���������d*
dtype0�
+model_2/embedding/embedding_lookup/IdentityIdentity+model_2/embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:���������d{
9model_2/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'model_2/global_average_pooling1d_1/MeanMean4model_2/embedding/embedding_lookup/Identity:output:0Bmodel_2/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:����������
%model_2/dense_5/MatMul/ReadVariableOpReadVariableOp.model_2_dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model_2/dense_5/MatMulMatMul0model_2/global_average_pooling1d_1/Mean:output:0-model_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&model_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_2/dense_5/BiasAddBiasAdd model_2/dense_5/MatMul:product:0.model_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
model_2/dense_5/ReluRelu model_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
%model_2/dense_6/MatMul/ReadVariableOpReadVariableOp.model_2_dense_6_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
model_2/dense_6/MatMulMatMul"model_2/dense_5/Relu:activations:0-model_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&model_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model_2/dense_6/BiasAddBiasAdd model_2/dense_6/MatMul:product:0.model_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� p
model_2/dense_6/ReluRelu model_2/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
%model_2/dense_7/MatMul/ReadVariableOpReadVariableOp.model_2_dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
model_2/dense_7/MatMulMatMul"model_2/dense_6/Relu:activations:0-model_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&model_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_2/dense_7/BiasAddBiasAdd model_2/dense_7/MatMul:product:0.model_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
model_2/dense_7/SigmoidSigmoid model_2/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentitymodel_2/dense_7/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^model_2/dense_5/BiasAdd/ReadVariableOp&^model_2/dense_5/MatMul/ReadVariableOp'^model_2/dense_6/BiasAdd/ReadVariableOp&^model_2/dense_6/MatMul/ReadVariableOp'^model_2/dense_7/BiasAdd/ReadVariableOp&^model_2/dense_7/MatMul/ReadVariableOp#^model_2/embedding/embedding_lookupQ^model_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 2P
&model_2/dense_5/BiasAdd/ReadVariableOp&model_2/dense_5/BiasAdd/ReadVariableOp2N
%model_2/dense_5/MatMul/ReadVariableOp%model_2/dense_5/MatMul/ReadVariableOp2P
&model_2/dense_6/BiasAdd/ReadVariableOp&model_2/dense_6/BiasAdd/ReadVariableOp2N
%model_2/dense_6/MatMul/ReadVariableOp%model_2/dense_6/MatMul/ReadVariableOp2P
&model_2/dense_7/BiasAdd/ReadVariableOp&model_2/dense_7/BiasAdd/ReadVariableOp2N
%model_2/dense_7/MatMul/ReadVariableOp%model_2/dense_7/MatMul/ReadVariableOp2H
"model_2/embedding/embedding_lookup"model_2/embedding/embedding_lookup2�
Pmodel_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2Pmodel_2/text_vectorization_1/string_lookup_1/hash_table_Lookup/LookupTableFindV2:M I
#
_output_shapes
:���������
"
_user_specified_name
examples:,(
&
_user_specified_nametable_handle:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_user_specified_name55132:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
R
8__inference_transform_features_layer_layer_call_fn_55307
text
identity�
PartitionedCallPartitionedCalltext*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_55302`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:M I
'
_output_shapes
:���������

_user_specified_nametext
�
]
#__inference_signature_wrapper_55035

inputs	
inputs_1
identity	

identity_1�
PartitionedCallPartitionedCallinputsinputs_1*
Tin
2	*
Tout
2	*:
_output_shapes(
&:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *!
fR
__inference_pruned_55027`
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:���������b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1
�

�
B__inference_dense_7_layer_call_and_return_conditional_losses_55415

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
__inference_save_fn_55720
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	��?MutableHashTable_lookup_table_export_values/LookupTableExportV2�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: �

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: �

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:d
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key:,(
&
_user_specified_nametable_handle"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
examples-
serving_default_examples:0���������<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
		tft_layer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses

2kernel
3bias"
_tf_keras_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
$B _saved_model_loader_tracked_dict"
_tf_keras_model
Q
1
*2
+3
24
35
:6
;7"
trackable_list_wrapper
Q
0
*1
+2
23
34
:5
;6"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Htrace_0
Itrace_12�
'__inference_model_2_layer_call_fn_55523
'__inference_model_2_layer_call_fn_55550�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0zItrace_1
�
Jtrace_0
Ktrace_12�
B__inference_model_2_layer_call_and_return_conditional_losses_55422
B__inference_model_2_layer_call_and_return_conditional_losses_55496�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0zKtrace_1
�
L	capture_1
M	capture_2
N	capture_3B�
 __inference__wrapped_model_55270text_xf"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zL	capture_1zM	capture_2zN	capture_3
�
O
_variables
P_iterations
Q_learning_rate
R_index_dict
S
_momentums
T_velocities
U_update_step_xla"
experimentalOptimizer
,
Vserving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
L
W	keras_api
Xlookup_table
Ytoken_counts"
_tf_keras_layer
�
Ztrace_02�
__inference_adapt_step_54449�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
`trace_02�
)__inference_embedding_layer_call_fn_55596�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0
�
atrace_02�
D__inference_embedding_layer_call_and_return_conditional_losses_55604�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
':%	�N2embedding/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
gtrace_02�
:__inference_global_average_pooling1d_1_layer_call_fn_55609�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zgtrace_0
�
htrace_02�
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_55615�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
ntrace_02�
'__inference_dense_5_layer_call_fn_55624�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0
�
otrace_02�
B__inference_dense_5_layer_call_and_return_conditional_losses_55635�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0
 :@2dense_5/kernel
:@2dense_5/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
utrace_02�
'__inference_dense_6_layer_call_fn_55644�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zutrace_0
�
vtrace_02�
B__inference_dense_6_layer_call_and_return_conditional_losses_55655�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zvtrace_0
 :@ 2dense_6/kernel
: 2dense_6/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
|trace_02�
'__inference_dense_7_layer_call_fn_55664�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z|trace_0
�
}trace_02�
B__inference_dense_7_layer_call_and_return_conditional_losses_55675�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z}trace_0
 : 2dense_7/kernel
:2dense_7/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
8__inference_transform_features_layer_layer_call_fn_55307�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_55302�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	_imported
�_wrapped_function
�_structured_inputs
�_structured_outputs
�_output_to_inputs_map"
trackable_dict_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
L	capture_1
M	capture_2
N	capture_3B�
'__inference_model_2_layer_call_fn_55523text_xf"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zL	capture_1zM	capture_2zN	capture_3
�
L	capture_1
M	capture_2
N	capture_3B�
'__inference_model_2_layer_call_fn_55550text_xf"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zL	capture_1zM	capture_2zN	capture_3
�
L	capture_1
M	capture_2
N	capture_3B�
B__inference_model_2_layer_call_and_return_conditional_losses_55422text_xf"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zL	capture_1zM	capture_2zN	capture_3
�
L	capture_1
M	capture_2
N	capture_3B�
B__inference_model_2_layer_call_and_return_conditional_losses_55496text_xf"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zL	capture_1zM	capture_2zN	capture_3
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
�
P0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
L	capture_1
M	capture_2
N	capture_3B�
#__inference_signature_wrapper_55188examples"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zL	capture_1zM	capture_2zN	capture_3
"
_generic_user_object
j
�_initializer
�_create_resource
�_initialize
�_destroy_resourceR jtf.StaticHashTable
T
�_create_resource
�_initialize
�_destroy_resourceR Z
table��
�
�	capture_1B�
__inference_adapt_step_54449iterator"�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_1
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
�B�
)__inference_embedding_layer_call_fn_55596inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_embedding_layer_call_and_return_conditional_losses_55604inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
:__inference_global_average_pooling1d_1_layer_call_fn_55609inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_55615inputs"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
'__inference_dense_5_layer_call_fn_55624inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_5_layer_call_and_return_conditional_losses_55635inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
'__inference_dense_6_layer_call_fn_55644inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_6_layer_call_and_return_conditional_losses_55655inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
'__inference_dense_7_layer_call_fn_55664inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_7_layer_call_and_return_conditional_losses_55675inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
8__inference_transform_features_layer_layer_call_fn_55307text"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_55302text"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�created_variables
�	resources
�trackable_objects
�initializers
�assets
�
signatures
$�_self_saveable_object_factories
�transform_fn"
_generic_user_object
0B.
__inference_pruned_55027inputsinputs_1
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
,:*	�N2Adam/m/embedding/embeddings
,:*	�N2Adam/v/embedding/embeddings
%:#@2Adam/m/dense_5/kernel
%:#@2Adam/v/dense_5/kernel
:@2Adam/m/dense_5/bias
:@2Adam/v/dense_5/bias
%:#@ 2Adam/m/dense_6/kernel
%:#@ 2Adam/v/dense_6/kernel
: 2Adam/m/dense_6/bias
: 2Adam/v/dense_6/bias
%:# 2Adam/m/dense_7/kernel
%:# 2Adam/v/dense_7/kernel
:2Adam/m/dense_7/bias
:2Adam/v/dense_7/bias
"
_generic_user_object
�
�trace_02�
__inference__creator_55679�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__initializer_55686�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__destroyer_55690�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__creator_55694�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__initializer_55698�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__destroyer_55702�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
!J	
Const_2jtf.TrackableConstant
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
�B�
__inference__creator_55679"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�
�	capture_1
�	capture_2B�
__inference__initializer_55686"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�	capture_1z�	capture_2
�B�
__inference__destroyer_55690"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__creator_55694"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__initializer_55698"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__destroyer_55702"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
#__inference_signature_wrapper_55035inputsinputs_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
�B�
__inference_save_fn_55720checkpoint_key"�
���
FullArgSpec
args�
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�	
� 
�B�
__inference_restore_fn_55727restored_tensors_0restored_tensors_1"�
���
FullArgSpec7
args/�,
jrestored_tensors_0
jrestored_tensors_1
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
	�	?
__inference__creator_55679!�

� 
� "�
unknown ?
__inference__creator_55694!�

� 
� "�
unknown A
__inference__destroyer_55690!�

� 
� "�
unknown A
__inference__destroyer_55702!�

� 
� "�
unknown J
__inference__initializer_55686(X���

� 
� "�
unknown C
__inference__initializer_55698!�

� 
� "�
unknown �
 __inference__wrapped_model_55270rXLMN*+23:;0�-
&�#
!�
text_xf���������
� "1�.
,
dense_7!�
dense_7���������j
__inference_adapt_step_54449JY�?�<
5�2
0�-�
����������IteratorSpec 
� "
 �
B__inference_dense_5_layer_call_and_return_conditional_losses_55635c*+/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������@
� �
'__inference_dense_5_layer_call_fn_55624X*+/�,
%�"
 �
inputs���������
� "!�
unknown���������@�
B__inference_dense_6_layer_call_and_return_conditional_losses_55655c23/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0��������� 
� �
'__inference_dense_6_layer_call_fn_55644X23/�,
%�"
 �
inputs���������@
� "!�
unknown��������� �
B__inference_dense_7_layer_call_and_return_conditional_losses_55675c:;/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
'__inference_dense_7_layer_call_fn_55664X:;/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
D__inference_embedding_layer_call_and_return_conditional_losses_55604f/�,
%�"
 �
inputs���������d	
� "0�-
&�#
tensor_0���������d
� �
)__inference_embedding_layer_call_fn_55596[/�,
%�"
 �
inputs���������d	
� "%�"
unknown���������d�
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_55615�I�F
?�<
6�3
inputs'���������������������������

 
� "5�2
+�(
tensor_0������������������
� �
:__inference_global_average_pooling1d_1_layer_call_fn_55609wI�F
?�<
6�3
inputs'���������������������������

 
� "*�'
unknown�������������������
B__inference_model_2_layer_call_and_return_conditional_losses_55422uXLMN*+23:;8�5
.�+
!�
text_xf���������
p

 
� ",�)
"�
tensor_0���������
� �
B__inference_model_2_layer_call_and_return_conditional_losses_55496uXLMN*+23:;8�5
.�+
!�
text_xf���������
p 

 
� ",�)
"�
tensor_0���������
� �
'__inference_model_2_layer_call_fn_55523jXLMN*+23:;8�5
.�+
!�
text_xf���������
p

 
� "!�
unknown����������
'__inference_model_2_layer_call_fn_55550jXLMN*+23:;8�5
.�+
!�
text_xf���������
p 

 
� "!�
unknown����������
__inference_pruned_55027�r�o
h�e
c�`
/
label&�#
inputs_label���������	
-
text%�"
inputs_text���������
� "a�^
.
label_xf"�
label_xf���������	
,
text_xf!�
text_xf����������
__inference_restore_fn_55727bYK�H
A�>
�
restored_tensors_0
�
restored_tensors_1	
� "�
unknown �
__inference_save_fn_55720�Y&�#
�
�
checkpoint_key 
� "���
u�r

name�
tensor_0_name 
*

slice_spec�
tensor_0_slice_spec 
$
tensor�
tensor_0_tensor
u�r

name�
tensor_1_name 
*

slice_spec�
tensor_1_slice_spec 
$
tensor�
tensor_1_tensor	�
#__inference_signature_wrapper_55035�i�f
� 
_�\
*
inputs �
inputs���������	
.
inputs_1"�
inputs_1���������"a�^
.
label_xf"�
label_xf���������	
,
text_xf!�
text_xf����������
#__inference_signature_wrapper_55188}XLMN*+23:;9�6
� 
/�,
*
examples�
examples���������"3�0
.
output_0"�
output_0����������
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_55302�:�7
0�-
+�(
&
text�
text���������
� "D�A
:�7
5
text_xf*�'
tensor_0_text_xf���������
� �
8__inference_transform_features_layer_layer_call_fn_55307o:�7
0�-
+�(
&
text�
text���������
� "1�.
,
text_xf!�
text_xf���������