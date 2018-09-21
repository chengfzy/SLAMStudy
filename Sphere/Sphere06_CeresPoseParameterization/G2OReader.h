#pragma once
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "glog/logging.h"

// read a single pose from the input and inserts it into the map, return false if there is a duplicate entry
template <typename Pose, typename Allocator>
bool readVertex(std::istream& fs, std::map<int, Pose, std::less<int>, Allocator>& poses) {
    int id;
    Pose pose;
    fs >> id >> pose;

    // ensure we don't have duplicate poses
    if (poses.count(id) == 1) {
        LOG(ERROR) << "Duplicate vertex with ID = " << id;
        return false;
    }

    poses[id] = pose;
    return true;
}

// read the constraints between two vertices in the pose graph
template <typename Constraint, typename Allocator>
void readConstraints(std::istream& fs, std::vector<Constraint, Allocator>& constraints) {
    Constraint constraint;
    fs >> constraint;

    constraints.emplace_back(constraint);
}

/**
 * Reads a file in the g2o filename format that describes a pose graph problem. The g2o format consists of two entries,
 * vertices and constraints.
 *
 * In 2D, a vertex is defined as follows:
 *  VERTEX_SE2 ID x_meters y_meters yaw_radians
 *
 * A constraint is defined as follows:
 *  EDGE_SE2 ID_A ID_B A_x_B A_y_B A_yaw_B I_11 I_12 I_13 I_22 I_23 I_33
 * where I_ij is the (i, j)-th entry of the information matrix for the measurement. Only the upper-triangular part is
 * stored
 *
 *
 * In 3D, a vertex is defined as follows:
 *  VERTEX_SE3:QUAT ID x y z q_x q_y q_z q_w
 * where the quaternion is in Hamilton form.
 *
 * A constraint is defined as follows:
 *  EDGE_SE3:QUAT ID_a ID_b x_ab y_ab z_ab q_x_ab q_y_ab q_z_ab q_w_ab I_11 I_12 I_13 .. I_16 I_22 I_23 .. I_26 .. I_66
 * where I_ij is the (i, j)-th entry of the information matrix for the measurement. Only the upper-triangular part is
 * stored. The measurement order is the delta position followed by the delta orientation.
 */
template <typename Pose, typename Constraint, typename MapAllocator, typename VectorAllocator>
bool readG2OFile(const std::string& filename, std::map<int, Pose, std::less<int>, MapAllocator>& poses,
                 std::vector<Constraint, VectorAllocator>& constraints) {
    poses.clear();
    constraints.clear();

    // read file
    std::ifstream fs(filename);
    if (!fs.is_open()) {
        LOG(ERROR) << "cannot open file \"" << filename << "\"";
        return false;
    }

    std::string dataType;
    while (fs.good()) {
        // read whether the type is a node or a constraint
        fs >> dataType;
        if (dataType == Pose::name()) {
            if (!readVertex(fs, poses)) {
                return false;
            }
        } else if (dataType == Constraint::name()) {
            readConstraints(fs, constraints);
        } else {
            LOG(ERROR) << "unknown data type: " << dataType;
            return false;
        }

        // clear any trailing whitespace from the line
        fs >> std::ws;
    }
    return true;
}