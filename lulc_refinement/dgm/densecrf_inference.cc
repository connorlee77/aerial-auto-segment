#include "DGM.h"
#include "VIS.h"
#include "DGM/timer.h"
#include "gdal_priv.h"

#include <errno.h>

using namespace DirectGraphicalModels;
using namespace DirectGraphicalModels::vis;

int main(int argc, char *argv[])
{
	
	
	const uint8_t nStates = 7; // {road, traffic island, grass, agriculture, tree, car}
	const uint16_t nFeatures = 3;

	const char *pszFilename = argv[1];

	GDALAllRegister();
	const GDALAccess eAccess = GA_ReadOnly;
	GDALDatasetUniquePtr poDataset = GDALDatasetUniquePtr(GDALDataset::FromHandle(GDALOpen(pszFilename, eAccess)));

	const int width = poDataset->GetRasterXSize();
	const int height = poDataset->GetRasterYSize();
	const int nBands = poDataset->GetRasterCount();
	const Size imgSize = Size(width, height);

	// std::cout << "Driver: " << poDataset->GetDriver()->GetDescription() << "/" << poDataset->GetDriver()->GetMetadataItem(GDAL_DMD_LONGNAME) << std::endl;
	// std::cout << "Size is " << poDataset->GetRasterXSize() << "x" << poDataset->GetRasterYSize() << "x" << poDataset->GetRasterCount() << std::endl;
	// std::cout << "Projection is " << poDataset->GetProjectionRef() << std::endl;

	// Reading parameters and images
	// Mat train_fv = imread(argv[1], 1);
	// resize(train_fv, train_fv, imgSize, 0, 0, INTER_LANCZOS4);	// training image feature vector
	// Mat train_gt = imread(argv[2], 0); resize(train_gt, train_gt, imgSize, 0, 0, INTER_NEAREST);	// groundtruth for training
	// Mat test_fv  = imread(argv[3], 1); resize(test_fv,  test_fv,  imgSize, 0, 0, INTER_LANCZOS4);	// testing image feature vector
	// Mat test_gt  = imread(argv[4], 0); resize(test_gt,  test_gt,  imgSize, 0, 0, INTER_NEAREST);	// groundtruth for evaluation
	// Mat test_img = imread(argv[5], 1); resize(test_img, test_img, imgSize, 0, 0, INTER_LANCZOS4);	// testing image

	// auto	nodeTrainer = CTrainNode::create(Bayes, nStates, nFeatures);
	// auto	graphKit	= CGraphKit::create(GraphType::dense, nStates);
	// CMarker	marker(DEF_PALETTE_6);
	// CCMat	confMat(nStates);

	// // ========================= STAGE 2: Training =========================
	// Timer::start("Training... ");
	// nodeTrainer->addFeatureVecs(train_fv, train_gt);
	// nodeTrainer->train();
	// Timer::stop();

	// // ==================== STAGE 3: Filling the Graph =====================
	// Timer::start("Filling the Graph... ");
	// Mat nodePotentials = nodeTrainer->getNodePotentials(test_fv);				// Classification: CV_32FC(nStates) <- CV_8UC(nFeatures)
	// graphKit->getGraphExt().setGraph(nodePotentials);							// Filling-in the graph nodes
	// graphKit->getGraphExt().addDefaultEdgesModel(100.0f, 3.0f);
	// graphKit->getGraphExt().addDefaultEdgesModel(test_fv, 300.0f, 10.0f);
	// Timer::stop();

	// // ========================= STAGE 4: Decoding =========================
	// Timer::start("Decoding... ");
	// vec_byte_t optimalDecoding = graphKit->getInfer().decode(100);
	// Timer::stop();

	// // ====================== Evaluation =======================
	// Mat solution(imgSize, CV_8UC1, optimalDecoding.data());
	// confMat.estimate(test_gt, solution);
	// char str[255];
	// sprintf(str, "Accuracy = %.2f%%", confMat.getAccuracy());
	// printf("%s\n", str);

	// // ====================== Visualization =======================
	// marker.markClasses(test_img, solution);
	// rectangle(test_img, Point(width - 160, height - 18), Point(width, height), CV_RGB(0, 0, 0), -1);
	// putText(test_img, str, Point(width - 155, height - 5), FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(225, 240, 255), 1, LineTypes::LINE_AA);
	// imwrite(argv[6], test_img);

	// imshow("Image", test_img);

	// waitKey();

	return 0;
}