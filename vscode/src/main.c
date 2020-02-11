#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <libv4l2.h>
#include <math.h>
#include <stdbool.h>

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define debug(msg) printf("debug: %s\n", msg)
char debugMsg[100];

struct buffer
{
    void *start;
    size_t length;
    size_t width;
    size_t height;
};
struct rgb
{
    uint8_t *r;
    uint8_t *g;
    uint8_t *b;
    int lenght;
};

#define RED 0
#define GREEN 1
#define BLUE 2

struct rgb createRGB(const struct buffer *image)
{
    struct rgb rgb_image;
    char *tab_image = image->start;
    int j = 0;
    rgb_image.r = malloc(image->length);
    rgb_image.g = malloc(image->length);
    rgb_image.b = malloc(image->length);
    rgb_image.lenght = image->length / 3;
    //explode in 3 colors array
    for (size_t i = 0, j = 0; i < (image->length); i += 3, j++)
    {
        *(rgb_image.r + j) = *(tab_image + i);
        *(rgb_image.g + j) = *(tab_image + i + 1);
        *(rgb_image.b + j) = *(tab_image + i + 2);
    }
    return rgb_image;
}
void grey_scale(struct buffer *image, struct rgb rgb_image)
{
    char *tab_image = image->start;
    int j;
    for (size_t i = 0, j = 0; i < (image->length); i += 3, j++)
    {
        float value = rgb_image.r[j] / 255.0 * 0.2126 + rgb_image.g[j] / 255.0 * 0.7152 + rgb_image.b[j] / 255.0 * .0722;
        if (value <= 0.0031308)
            value *= 12.92;
        else
        {
            value = 1.055 * pow(value, 1 / 2.4) - 0.055;
        }

        *(tab_image + i) = value * 255;
        *(tab_image + i + 1) = value * 255;
        *(tab_image + i + 2) = value * 255;
    }
}
void rgb_filtre(struct buffer *image, struct rgb rgb_image, bool r, bool g, bool b)
{
    char *tab_image = image->start;
    memset(tab_image, 0, image->length);
    int j;
    for (size_t i = 0, j = 0; i < (image->length); i += 3, j++)
    {
        if (r)
            *(tab_image + i) = *(rgb_image.r + j);
        if (g)
            *(tab_image + i + 1) = *(rgb_image.g + j);
        if (b)
            *(tab_image + i + 2) = *(rgb_image.b + j);
    }
}

void traitement(struct buffer *image)
{
    struct rgb rgb_image;
    rgb_image = createRGB(image);
    grey_scale(image, rgb_image);
    //rgb_filtre(image, rgb_image, false, true, false);
}

static void xioctl(int fh, int request, void *arg)
{
    int r;
    do
    {
        r = v4l2_ioctl(fh, request, arg);
    } while (r == -1 && ((errno == EINTR) || (errno == EAGAIN)));
    if (r == -1)
    {
        fprintf(stderr, "error %d, %s\\n", errno, strerror(errno));
        exit(EXIT_FAILURE);
    }
}
int waitImage(int fd)
{
    fd_set fds;
    struct timeval tv;
    int r;
    do
    {
        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        tv.tv_sec = 2;
        tv.tv_usec = 0;
        r = select(fd + 1, &fds, NULL, NULL, &tv);
    } while ((r == -1 && (errno = EINTR)));
    return r;
}

void saveppm(const char *file_name, struct buffer image)
{
    FILE *fout;
    fout = fopen(file_name, "w");
    if (!fout)
    {
        perror("Cannot open image");
        exit(EXIT_FAILURE);
    }
    puts("saving file ppm");
    fprintf(fout, "P6\n%lu %lu 255\n",
            image.width, image.height);
    fwrite(image.start, image.length, 1, fout);
    fclose(fout);
}

void savejpg(const char *file_name, struct buffer image)
{
    int jpgfile;
    if ((jpgfile = open(file_name, O_WRONLY | O_CREAT, 0660)) < 0)
    {
        perror("open");
        exit(1);
    }
    write(jpgfile, image.start, image.length);
    close(jpgfile);
}

void printCapabilities(int fd, const char *dev_name)
{
    //get capabilities
    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0)
    {
        perror("VIDIOC_QUERYCAP");
        exit(1);
    }
    printf("driver : %s\ncard : %s \nbus_info:%s\n ", cap.driver, cap.card, cap.bus_info);
    printf("Version: %u.%u.%u\n",

           (cap.version >> 16) & 0xFF, (cap.version >> 8) & 0xFF, cap.version & 0xFF);
    printf("capabilities %X\n", cap.capabilities);
    if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) //obligatoire
        printf("The device supports the single-planar API through the Video Capture\n");
    if (cap.capabilities & V4L2_CAP_EXT_PIX_FORMAT)
        printf("The device supports the struct v4l2_pix_format extended fields.\n");
    if (cap.capabilities & V4L2_CAP_META_CAPTURE)
        printf("The device supports the Metadata Interface capture interface.\n");
    if (cap.capabilities & V4L2_CAP_STREAMING) // obligatoire
        printf("The device supports the streaming I/O method.\n");
    if (cap.capabilities & V4L2_CAP_DEVICE_CAPS)
    {
        printf("devices_cap=%X\n", cap.device_caps);
    }
    if (!(cap.capabilities & V4L2_CAP_READWRITE))
        printf("%s does not support read i/o\n", dev_name);
    puts("list of availables formats :");
    struct v4l2_fmtdesc fmtdesc;
    memset(&fmtdesc, 0, sizeof(fmtdesc));
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0)
    {
        printf("%s\n", fmtdesc.description);
        fmtdesc.index++;
    }
}

struct buffer initCapture(int fd, int width, int height)
{
    struct v4l2_format format;
    struct buffer image;
    struct v4l2_buffer buf;
    CLEAR(format);
    debug("change TO RGB FORMAT");
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = width;
    format.fmt.pix.height = height;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    format.fmt.pix.field = V4L2_FIELD_INTERLACED;
    xioctl(fd, VIDIOC_S_FMT, &format);
    if (format.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB24)
    {
        printf("Libv4l didn't accept RGB24 format. Can't proceed.\\n");
        exit(EXIT_FAILURE);
    }
    if ((format.fmt.pix.width != 640) || (format.fmt.pix.height != 480))
        printf("Warning: driver is sending image at %dx%d\\n",
               format.fmt.pix.width, format.fmt.pix.height);

    struct v4l2_requestbuffers req;
    CLEAR(req);
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(fd, VIDIOC_REQBUFS, &req);
    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    xioctl(fd, VIDIOC_QUERYBUF, &buf);
    image.length = buf.length;
    image.start = v4l2_mmap(NULL, buf.length,
                            PROT_READ | PROT_WRITE, MAP_SHARED,
                            fd, buf.m.offset);
    if (MAP_FAILED == image.start)
    {
        perror("mmap");
        exit(EXIT_FAILURE);
    }
    image.width = format.fmt.pix.width;
    image.height = format.fmt.pix.height;
    sprintf(debugMsg, "image width=%d, height=%d\n", image.width, image.height);
    debug(debugMsg);
    return image;
}
void captureImage(int fd, struct buffer image)
{
    struct v4l2_buffer buf;
    enum v4l2_buf_type type;
    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    xioctl(fd, VIDIOC_QBUF, &buf);
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    debug("STREAMON");
    xioctl(fd, VIDIOC_STREAMON, &type);
    waitImage(fd);
    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    xioctl(fd, VIDIOC_DQBUF, &buf);
    xioctl(fd, VIDIOC_QBUF, &buf);
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    debug("STREAMOFF");
    xioctl(fd, VIDIOC_STREAMOFF, &type);
}

int main(int argc, char **argv)
{
    int fd = -1;
    struct buffer image;
    const char *dev_name = "/dev/video0";
    fd = v4l2_open(dev_name, O_RDWR | O_NONBLOCK, 0);
    if (fd < 0)
    {
        perror("Cannot open device");
        exit(EXIT_FAILURE);
    }
#ifdef debug
    printCapabilities(fd, dev_name);
#endif
    image = initCapture(fd, 640, 480);
    captureImage(fd, image);
    debug("SAVING TO in.ppm");
    saveppm("in.ppm", image);
    traitement(&image);
    debug("SAVING TO out.ppm");
    saveppm("out.ppm", image);
    v4l2_munmap(image.start, image.length);
    v4l2_close(fd);
    return 0;
}
