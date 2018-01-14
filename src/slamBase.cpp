
#include "slamBase.h"
PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
PointCloud::Ptr cloud ( new PointCloud );
pcl::PointXYZRGBA cloud_point;
double d;
for(int r = 0; r < depth.rows; r++)   
    for(int c= 0;c< depth.cols;c++)
     {
        d= double(depth.ptr<ushort>(r)[c]);
        if (d==0)
        continue;
        CAMERA_INTRINSIC_PARAMETERS *p;
        p = &camera;     
        cloud_point.z= d/(p->scale);
        cloud_point.x= (c-p->cx)*cloud_point.z/p->fx;
        cloud_point.y= (r-p->cy)*cloud_point.z/p->fy; 
        cv::Vec3b* ptr = rgb.ptr<cv::Vec3b>(r);
        cloud_point.b=ptr[c][0];
        cloud_point.g=ptr[c][1];
        cloud_point.r=ptr[c][2];      
        cloud->points.push_back(cloud_point);        
    }
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
    return cloud;
}

cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f *p1,*p2,cloud_point;
    p1=&cloud_point;
    p2=&point;
    CAMERA_INTRINSIC_PARAMETERS *p;
    p = &camera;     
    p1->z= p2->z/(p->scale);
    p1->x= (p2->x-p->cx)*p1->z/p->fx;
    p1->y= (p2->y-p->cy)*p1->z/p->fy; 
    return cloud_point;
}
void computeKeyPointsAndDesp( FRAME& frame, string detector_name, string descriptor_name )
{
    cv::Ptr<cv::FeatureDetector>  detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor;
    detector = cv::FeatureDetector::create( detector_name.c_str() );
    descriptor = cv::DescriptorExtractor::create( descriptor_name.c_str() );
    detector->detect( frame.rgb, frame.kp );
    descriptor->compute( frame.rgb, frame.kp, frame.desp );
    return;
}
 vector<cv::DMatch> matches_desp(FRAME& frame1, FRAME& frame2)
{ 
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher;
    matcher.match( frame1.desp, frame2.desp, matches );
    return matches;
}

vector<cv::DMatch> matches_optimize(vector<cv::DMatch> matches)
{
    vector<cv::DMatch> opt_matches;
    long int i;
    double dmin;
    dmin=matches[0].distance;  
    for(i=0;i<matches.size();i++)
    {
        if (matches[i].distance<dmin)
        dmin=matches[i].distance;
    }
    for(i=0;i<matches.size();i++ )
    {
        if (matches[i].distance < 4*dmin )
        opt_matches.push_back(matches[i]);
    }
    return opt_matches;
}

RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    vector< cv::DMatch > matches,goodMatches;
    vector<cv::Point3f> point_3D;
    vector< cv::Point2f > point_2D;
    matches =matches_desp(frame1,frame2);  
    goodMatches=matches_optimize(matches);
    RESULT_OF_PNP motion;
    long int i=0;
    double d;
    
    for(i=0; i<goodMatches.size(); i++)
    {
       
        cv::KeyPoint point1 = frame1.kp[goodMatches[i].queryIdx];
        cv::KeyPoint point2 = frame2.kp[goodMatches[i].trainIdx];
        d=double(frame1.depth.ptr<ushort>(int(point1.pt.y))[int(point1.pt.x)]);
        cv::Point3f p1,P;
        cv::Point2f p2;   
        p2.x=point2.pt.x;
        p2.y=point2.pt.y;
        point_2D.push_back(p2);
        p1.x=point1.pt.x;
        p1.y=point1.pt.y;
        p1.z=d;
        P=point2dTo3d(p1,camera);
        point_3D.push_back(P);
    }
    if (point_3D.size() ==0 || point_2D.size()==0)
    {
        motion.inliers = -1;
        return motion;
    }
    CAMERA_INTRINSIC_PARAMETERS *p;
    p = &camera;
    double camera_matrix[3][3]={{p->fx, 0, p->cx},{0, p->fy, p->cy},{0, 0, 1}};
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix);
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(point_3D, point_2D, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );
    motion.rvec = rvec;
    motion.tvec = tvec;
    motion.inliers = inliers.rows;
    return motion;
}

// cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec )
{
    cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ ) 
            r(i,j) = R.at<double>(i,j);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0); 
    T(1,3) = tvec.at<double>(1,0); 
    T(2,3) = tvec.at<double>(2,0);
    return T;
}

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera ) 
{
    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgb, newFrame.depth, camera );
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;
    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridsize = atof( pd.getData("voxel_grid").c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter( *tmp );
    return tmp;
}

double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}

FRAME readFrame( int index, ParameterReader& pd )
{
    FRAME f;
    string rgbDir   =   pd.getData("rgb_dir");
    string depthDir =   pd.getData("depth_dir");
    
    string rgbExt   =   pd.getData("rgb_extension");
    string depthExt =   pd.getData("depth_extension");

    stringstream ss;
    ss<<rgbDir<<index<<rgbExt;
    string filename;
    ss>>filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;
    f.depth = cv::imread( filename, -1 );
    f.frameID = index;
    return f;
}