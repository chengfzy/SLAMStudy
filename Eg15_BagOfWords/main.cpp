#include <iostream>
#include <string>
#include <vector>
#include "DBoW3/DBoW3.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

DEFINE_string(imageFolder, "../../data/bow_images/", "image folder for DoW training");
DEFINE_string(saveVocFile, "./vocabulary.yml.gz", "save vocabulary file name");
DEFINE_string(importVocFile, "../../data/vocab_larger.yml.gz", "imported vocabulary file name");

// generate vocabulary from 10 images
void generateVoc() {
    LOG(INFO) << endl << "============================Generate Vocabulary from 10 Images ============================";
    // read images
    LOG(INFO) << "reading images...";
    vector<Mat> images;
    for (size_t i = 0; i < 10; ++i) {
        images.emplace_back(imread(FLAGS_imageFolder + to_string(i + 1) + ".png", IMREAD_UNCHANGED));
    }

    // detect ORB features
    LOG(INFO) << "detecting ORB features...";
    Ptr<Feature2D> detector = ORB::create();
    vector<Mat> descriptors;
    for (auto& m : images) {
        vector<KeyPoint> keyPoints;
        Mat desc;
        detector->detectAndCompute(m, noArray(), keyPoints, desc);
        descriptors.emplace_back(desc);
    }

    // create vocabulary
    LOG(INFO) << "create vocabulary...";
    DBoW3::Vocabulary voc;
    voc.create(descriptors);
    cout << "vocabulary info: " << voc << endl;
    voc.save(FLAGS_saveVocFile);
}

// loop closure to calculate the similarity
void loopClosure() {
    LOG(INFO) << endl << "========================Loop Closure to Compare Similarity ofImages========================";

    // read databases
    LOG(INFO) << "reading database";
    DBoW3::Vocabulary voc(FLAGS_importVocFile);
    if (voc.empty()) {
        LOG(FATAL) << "vocabulary does not exist";
        return;
    }

    // read images
    LOG(INFO) << "reading images";
    vector<Mat> images;
    for (size_t i = 0; i < 10; ++i) {
        images.emplace_back(imread(FLAGS_imageFolder + to_string(i + 1) + ".png", IMREAD_UNCHANGED));
    }

    // detect ORB features
    LOG(INFO) << "detecting ORB features...";
    Ptr<Feature2D> detector = ORB::create();
    vector<Mat> descriptors;
    for (auto& m : images) {
        vector<KeyPoint> keyPoints;
        Mat desc;
        detector->detectAndCompute(m, noArray(), keyPoints, desc);
        descriptors.emplace_back(desc);
    }

    // compare images directly
    LOG(INFO) << "comparing images with images";
    for (size_t i = 0; i < images.size(); ++i) {
        DBoW3::BowVector v1;
        voc.transform(descriptors[i], v1);
        for (size_t j = 0; j < images.size(); ++j) {
            DBoW3::BowVector v2;
            voc.transform(descriptors[j], v2);
            double score = voc.score(v1, v2);
            cout << "image " << i << " vs image " << j << " : " << score << endl;
        }
        cout << endl;
    }

    // or compare one image to a database images
    LOG(INFO) << "comparing images with database";
    DBoW3::Database db(voc, false, 0);
    for (size_t i = 0; i < descriptors.size(); ++i) {
        db.add(descriptors[i]);
    }
    cout << "database info: " << db << endl;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        DBoW3::QueryResults ret;
        db.query(descriptors[i], ret, 4);  // max result = 4
        cout << "searching for image " << i << " return " << ret << endl;
    }
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_alsologtostderr = true;

    generateVoc();
    loopClosure();

    google::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
    return 0;
}
