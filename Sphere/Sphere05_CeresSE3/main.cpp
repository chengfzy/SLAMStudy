#include "G2OReader.h"
#include "PoseGraph3DErrorTerm.h"
#include "SE3Parameterization.h"
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "types.h"

using namespace ceres;
using namespace std;

DEFINE_string(inputFile, "../../data/sphere2500.g2o", "pose graph definition filename in g2o format");

// save poses to the file with format: ID p_x p_y p_z q_x q_y q_z q_w
bool savePose(const string& filename, const MapOfPoses& poses) {
    fstream fs(filename, ios::out);
    if (!fs.is_open()) {
        LOG(ERROR) << "cannot create file \"" << filename << "\"";
        return false;
    }
    for (auto& p : poses) {
        fs << p.first << " " << p.second.T.translation().transpose() << " " << p.second.T.so3().params().x() << " "
           << p.second.T.so3().params().y() << " " << p.second.T.so3().params().z() << " "
           << p.second.T.so3().params().w() << endl;
    }
    fs.close();
    return true;
}

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // check input
    CHECK(!FLAGS_inputFile.empty()) << "need to specify the input filename";

    // read data from file
    MapOfPoses poses;
    VectorOfConstaints constraints;
    if (!readG2OFile(FLAGS_inputFile, poses, constraints)) {
        LOG(FATAL) << "read data from file failed";
    }
    cout << "number of poses: " << poses.size() << endl;
    cout << "number of constraints: " << constraints.size() << endl;

    // save original poses
    savePose("./sphere_original.txt", poses);

    // build problem
    Problem problem;
    LossFunction* lossFunction = nullptr;
    LocalParameterization* se3Parameterization = new SE3Parameterization;
    for (auto& c : constraints) {
        auto itPoseBegin = poses.find(c.idBegin);
        CHECK(itPoseBegin != poses.end()) << "Pose with ID = " << c.idBegin << " not found";
        auto itPoseEnd = poses.find(c.idEnd);
        CHECK(itPoseEnd != poses.end()) << "Pose with ID = " << c.idEnd << " not found";

        const Eigen::Matrix<double, 6, 6> sqrtInformation = c.information.llt().matrixL();
        CostFunction* costFunction = PoseGraph3DErrorTerm::create(c.t_be, sqrtInformation);
        problem.AddResidualBlock(costFunction, lossFunction, itPoseBegin->second.T.data(), itPoseEnd->second.T.data());
        problem.SetParameterization(itPoseBegin->second.T.data(), se3Parameterization);
        problem.SetParameterization(itPoseEnd->second.T.data(), se3Parameterization);
    }
    // constrain the gauge freedom by setting one of the poses as constant so the optimizer cannot change it
    auto itPoseStart = poses.begin();
    CHECK(itPoseStart != poses.end()) << "There are no poses";
    cout << "start poses: p = " << itPoseStart->second.T.translation().transpose()
         << ", q = " << itPoseStart->second.T.so3().params().transpose() << endl;
    problem.SetParameterBlockConstant(itPoseStart->second.T.data());

    // solve problem
    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;

    // save optimized poses
    savePose("./sphere_optimized.txt", poses);

    google::ShutdownGoogleLogging();
    google::ShutDownCommandLineFlags();
    return 0;
}