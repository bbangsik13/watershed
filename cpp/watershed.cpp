#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <cmath>
#include <queue>
#include <chrono>



/**************************[ queue 관련 ]*******************************/
struct WS_struct
{
public:
    float dist;
    int x;
    int y;
    int marker;


    WS_struct(float dist, int x, int y, int marker)
    {
        this->dist = dist;
        this->x = x;
        this->y = y;
        this->marker = marker;
    }
};


bool operator<(const WS_struct& a, const WS_struct& b)
{
    return a.dist > b.dist;
}

/**************************[ gredient 구현 ]*******************************/

cv::Mat gredient(cv::Mat arr) {
    int width = arr.cols;
    int height = arr.rows;

   
    cv::Mat diff_img = cv::Mat::zeros(width, height, CV_32F);
    /*
    // diff_img[X or Y 가 0인 부분] <= arr[X or Y]
    for (int x = 0; x < width; x++) {
        diff_img.at<float>(x, 0) = arr.at<uchar>(x, 0);
    }
    for (int y = 1; y < height; y++) {
        diff_img.at<float>(0, y) = arr.at<uchar>(0, y);
    }

    //diff_img [x] = root( (arr[x]-arr[x-1])^2 + (arr[y]-arr[y-1])^2 )
    for (int y = 1; y < height; y++) {
        for (int x = 1; x < width; x++) {
            diff_img.at<float> (x, y) =
                sqrtf(
                    powf((arr.at<uchar>(x,y) - arr.at<uchar>(x-1, y))/2.0, 2.0) +
                    powf((arr.at<uchar>(x,y) - arr.at<uchar>(x, y-1))/2.0, 2.0)
                );
        }
    }
    */


    // diff_img[X or Y 가 0인 부분] <= arr[X or Y]

    for (int x = 1; x < width-1; x++) {
        diff_img.at<float>(x, 0) =
            sqrtf(
                powf((arr.at<uchar>(x+1, 0) - arr.at<uchar>(x-1, 0))/2.0 , 2.0) +
                powf((arr.at<uchar>(x, 1) - arr.at<uchar>(x, 0)), 2.0)
            );
        diff_img.at<float>(x, height-1) =
            sqrtf(
                powf((arr.at<uchar>(x + 1, height-1) - arr.at<uchar>(x - 1, height-1)) / 2.0, 2.0) +
                powf((arr.at<uchar>(x, height-1) - arr.at<uchar>(x, height-2)), 2.0)
            );
    }
    for (int y = 1; y < height-1; y++) {
        diff_img.at<float>(0, y) =
            sqrtf(
                powf((arr.at<uchar>(1, y) - arr.at<uchar>(0, y)), 2.0) +
                powf((arr.at<uchar>(0, y + 1) - arr.at<uchar>(0, y - 1)) / 2.0, 2.0)
            );
        diff_img.at<float>(width-1, y) =
            sqrtf(
                powf((arr.at<uchar>(width-1, y) - arr.at<uchar>(width-2, y)), 2.0) +
                powf((arr.at<uchar>(width-1, y + 1) - arr.at<uchar>(width-1, y - 1)) / 2.0, 2.0)
            );
    }

    diff_img.at<float>(0, 0) = 
        sqrtf(
        powf((arr.at<uchar>(1,0) - arr.at<uchar>(0, 0)), 2.0) +
        powf((arr.at<uchar>(0,1) - arr.at<uchar>(0, 0)), 2.0)
    );
    diff_img.at<float>(0, height-1) =
        sqrtf(
            powf((arr.at<uchar>(1, height-1) - arr.at<uchar>(0, height-1)), 2.0) +
            powf((arr.at<uchar>(0, height-1) - arr.at<uchar>(0, height-2)), 2.0)
        );
    diff_img.at<float>(width-1, 0) =
        sqrtf(
            powf((arr.at<uchar>(width-1, 0) - arr.at<uchar>(width-2, 0)), 2.0) +
            powf((arr.at<uchar>(width-1, 1) - arr.at<uchar>(width-1, 0)), 2.0)
        );
    diff_img.at<float>(width-1, height-1) =
        sqrtf(
            powf((arr.at<uchar>(width - 1, height-1) - arr.at<uchar>(width - 2, height-1)), 2.0) +
            powf((arr.at<uchar>(width - 1, height-1) - arr.at<uchar>(width - 1, height-2)), 2.0)
        );

    //diff_img [x] = root( (arr[x]-arr[x-1])^2 + (arr[y]-arr[y-1])^2 )
    for (int y = 1; y < height-1; y++) {
        for (int x = 1; x < width-1; x++) {
            diff_img.at<float>(x, y) =
                sqrtf(
                    powf((arr.at<uchar>(x+1, y) - arr.at<uchar>(x - 1, y)) / 2.0, 2.0) +
                    powf((arr.at<uchar>(x, y+1) - arr.at<uchar>(x, y - 1)) / 2.0, 2.0)
                );
        }
    }




    /*
    cv::Mat Sobal_x;
    cv::Mat Sobal_y;

    cv::Sobel(arr, Sobal_x, CV_8UC1, 1, 0, 3, 1, 128);
    cv::Sobel(arr, Sobal_y, CV_8UC1, 0, 1, 3, 1, 128);
    diff_img = abs(Sobal_x-128) + abs(Sobal_y-128);
    */

    return diff_img;
}

/*************************[ add함수(queue push)구현 ]********************************/

void addleft(std::priority_queue<WS_struct>& WS_que, int marker, int x, int y, cv::Mat img_grad, cv::Mat tag) {
    x = x - 1;
    if ( x >= 0 ) {
        if (!tag.at<uchar> (x,y)) {
            //queue 넣기 (dist, x, y, marker)
            WS_struct wst(img_grad.at<uchar>(x, y), x, y, marker);
            WS_que.push(wst);
        }
    }
}

void addright(std::priority_queue<WS_struct>& WS_que, int marker, int x, int y, cv::Mat img_grad, cv::Mat tag) {
    x = x + 1;
    if (x < img_grad.cols) {
        if (!tag.at<uchar> (x,y)) {
            //queue 넣기 (dist, x, y, marker)
            WS_struct wst(img_grad.at<uchar>(x, y), x, y, marker);
            WS_que.push(wst);
        }
    }
}

void addup(std::priority_queue<WS_struct>& WS_que, int marker, int x, int y, cv::Mat img_grad, cv::Mat tag) {
    y = y-1;
    if ( y >= 0 ) {
        if (!tag.at<uchar> (x,y)) {
            //queue 넣기 (dist, x, y, marker)
            WS_struct wst(img_grad.at<uchar>(x, y), x, y, marker);
            WS_que.push(wst);
        }
    }
}

void adddown(std::priority_queue<WS_struct>& WS_que, int marker, int x, int y, cv::Mat img_grad, cv::Mat tag) {
    y=y+1;
    if ( y < img_grad.rows ) {
        if (!tag.at<uchar> (x,y)) {
            //queue 넣기 (dist, x, y, marker)
            WS_struct wst(img_grad.at<uchar>(x, y), x, y, marker);
            WS_que.push(wst);
        }
    }
}



void f_addleft(std::priority_queue<WS_struct>& WS_que, int marker, int x, int y, cv::Mat img_grad, cv::Mat tag) {
    x = x - 1;
    if (x >= 0) {
        if (!tag.at<uchar>(x, y)) {
            //queue 넣기 (dist, x, y, marker)
            WS_struct wst(img_grad.at<float>(x, y), x, y, marker);
            WS_que.push(wst);
        }
    }
}

void f_addright(std::priority_queue<WS_struct>& WS_que, int marker, int x, int y, cv::Mat img_grad, cv::Mat tag) {
    x = x + 1;
    if (x < img_grad.cols) {
        if (!tag.at<uchar>(x, y)) {
            //queue 넣기 (dist, x, y, marker)
            WS_struct wst(img_grad.at<float>(x, y), x, y, marker);
            WS_que.push(wst);
        }
    }
}

void f_addup(std::priority_queue<WS_struct>& WS_que, int marker, int x, int y, cv::Mat img_grad, cv::Mat tag) {
    y = y - 1;
    if (y >= 0) {
        if (!tag.at<uchar>(x, y)) {
            //queue 넣기 (dist, x, y, marker)
            WS_struct wst(img_grad.at<float>(x, y), x, y, marker);
            WS_que.push(wst);
        }
    }
}

void f_adddown(std::priority_queue<WS_struct>& WS_que, int marker, int x, int y, cv::Mat img_grad, cv::Mat tag) {
    y = y + 1;
    if (y < img_grad.rows) {
        if (!tag.at<uchar>(x, y)) {
            //queue 넣기 (dist, x, y, marker)
            WS_struct wst(img_grad.at<float>(x, y), x, y, marker);
            WS_que.push(wst);
        }
    }
}

/*********************************************************/

//tag 빈배열

//S priority queue          

cv::Mat watershed(cv::Mat image, cv::Mat tag) {

    std::priority_queue<WS_struct> WS_que;
 


    cv::Mat img_grad = gredient(image);
    cv::namedWindow("Input Image");
    cv::imshow("Input Image", img_grad);
    cv::waitKey(0);

    //전처리..?
    
    //cv::GaussianBlur(img_grad, img_grad, cv::Size(3,3), 10.0);
    



    // 임시 초기 라벨링

    /**********/
    WS_struct red(img_grad.at<float>(306, 306), 306, 306, 80);
    WS_que.push(red);


    WS_struct black1(img_grad.at<float>(376, 225), 376, 225, 160);
    WS_struct black2(img_grad.at<float>(231, 394), 231, 394, 160);
    WS_struct black3(img_grad.at<float>(437, 386), 437, 386, 160);
    WS_que.push(black1);
    WS_que.push(black2);
    WS_que.push(black3);


    WS_struct white1(img_grad.at<float>(30, 30), 30, 30, 255);
    WS_struct white2(img_grad.at<float>(438, 310), 438, 310, 255);
    WS_struct white3(img_grad.at<float>(241, 317), 241, 317, 255);
    WS_struct white4(img_grad.at<float>(262, 474), 262, 474, 255);
    WS_que.push(white1);
    WS_que.push(white2);
    WS_que.push(white3);
    WS_que.push(white4);



    /**********/
    int test_cnt = 0;
    while (!WS_que.empty()) {//S에 있는 주변부 확장
        WS_struct wst = WS_que.top(); //class
        WS_que.pop();
        int x = wst.x;
        int y = wst.y;
        int marker = wst.marker;

        if (!tag.at<uchar>(x, y)) { //미정 tag marker로 마킹
            tag.at<uchar>(x, y) = marker;
            f_addleft(WS_que, marker, x, y, img_grad, tag);
            f_addright(WS_que, marker, x, y, img_grad, tag);
            f_addup(WS_que, marker, x, y, img_grad, tag);
            f_adddown(WS_que, marker, x, y, img_grad, tag);
        }
        /*
        if (!(test_cnt++ % 1000)) {
            cv::namedWindow("test");
            cv::imshow("test", tag);
            cv::waitKey(0);
            cv::destroyWindow("test");
        }*/
        
    }

    return tag;
}

int main(void)
{
    cv::Mat img_gray = cv::imread("test.jpg", cv::IMREAD_GRAYSCALE);
    cv::fastNlMeansDenoising(img_gray, img_gray, 10, 7, 21);
    cv::Mat tag = cv::Mat::zeros(img_gray.cols, img_gray.rows, CV_8UC1);


    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    watershed(img_gray, tag);

    std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
    std::cout << "Time cost(sec): " << sec.count() << "seconds" << std::endl;

    cv::namedWindow("Input Image");
    cv::imshow("Input Image", img_gray);
    cv::namedWindow("Output Image");
    cv::imshow("Output Image", tag);



    cv::waitKey(0);
    cv::destroyAllWindows();


    return 0;


}
