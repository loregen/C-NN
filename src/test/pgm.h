typedef struct Net_ Net;

typedef struct PGMData_{
    int row;
    int col;
    int max_gray;
    double *vector;
}PGMData;

void writePGMImage(const char *filename, PGMData *data);
void writeKernelsToPGM(Net *net, int layer);