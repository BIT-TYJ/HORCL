DIRECTORY=`dirname $0`

ROOTDIRECTORY=$DIRECTORY/..

$ROOTDIRECTORY/build/test/ceres/cauchy_em_no_sem $ROOTDIRECTORY/data/cauchy/odometry.txt $ROOTDIRECTORY/data/cauchy/loop_point.txt $ROOTDIRECTORY/data/cauchy/init.txt $ROOTDIRECTORY/data/cauchy/semantic_p_matrix.txt  $ROOTDIRECTORY/data/cauchy/poses_new_no_sem.txt $ROOTDIRECTORY/data/cauchy/keep_new.txt
