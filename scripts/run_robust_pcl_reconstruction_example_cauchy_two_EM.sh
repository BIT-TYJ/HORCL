DIRECTORY=`dirname $0`

ROOTDIRECTORY=$DIRECTORY/..

$ROOTDIRECTORY/build/test/ceres/cauchy_em_two_EM $ROOTDIRECTORY/data/cauchy/odometry.txt $ROOTDIRECTORY/data/cauchy/loop_point.txt $ROOTDIRECTORY/data/cauchy/init.txt $ROOTDIRECTORY/data/cauchy/semantic_p_matrix.txt  $ROOTDIRECTORY/data/cauchy/poses_two_EM.txt $ROOTDIRECTORY/data/cauchy/keep_new.txt
