// Regression test for the two Visualizer::publishMap changes (spade-pgo 1fb9387).
//
//   1. pcl::VoxelGrid -> pcl::ApproximateVoxelGrid
//   2. filter() into a separate cloud instead of in place
//
// Both are claims about runtime behaviour, so assert the behaviour rather than that it builds.
//
// History: change (1) was itself wrong and was replaced on 15 Aug 2026 by
// pcl::octree::OctreePointCloudVoxelCentroid, which is what publishMap uses now.
// pcl::ApproximateVoxelGrid merges only within a fixed 512-slot cache (histsize_, hardcoded in
// PCL 1.10) so it needs spatially coherent input; assembleGlobalPointCloud() concatenates
// keyframe by keyframe, so points sharing a voxel arrive ~100k apart and always miss:
//
//     pcl::VoxelGrid                1998 ms   1 203 416 pts    19.3 MB  (bails out on a big bbox)
//     pcl::ApproximateVoxelGrid      693 ms  11 621 194 pts   185.9 MB  <- barely filters
//     OctreePointCloudVoxelCentroid 2599 ms   1 204 086 pts    19.3 MB  <- shipped
//     no filter at all                       14 600 580 pts   233.6 MB
//
// Change (2), filtering into a separate cloud, is sound and was kept.
//
// Uses the same PointType (pcl::PointXYZI) and the same 0.3 m leaf as
// config/multiswarm_mid360.yaml, on real map geometry: XYZ dumped from
// 20241016_131631_map.las, recentred to the origin, as raw float32 triples.
//
//   python3 -c "import laspy,numpy as np; f=laspy.read('20241016_131631_map.las'); \
//     a=np.column_stack([f.x,f.y,f.z]); (a-a.min(0)).astype('f4').tofile('map_xyz.f32')"
//   g++ -O2 -std=c++14 -o tv test_voxelizer.cpp -I/usr/include/pcl-1.10 -I/usr/include/eigen3 \
//       -lpcl_common -lpcl_filters -lpcl_octree
//   ./tv map_xyz.f32

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

typedef pcl::PointXYZI PointType;

static const double LEAF = 0.3;   // params->visualize.voxel_size

static int failures = 0;

static void check(bool ok, const char *what)
{
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static pcl::PointCloud<PointType>::Ptr load(const char *path, double stray_m)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    size_t n = bytes / (3 * sizeof(float));
    std::vector<float> buf(n * 3);
    if (fread(buf.data(), sizeof(float), n * 3, fp) != n * 3) { fprintf(stderr, "short read\n"); exit(2); }
    fclose(fp);

    pcl::PointCloud<PointType>::Ptr c(new pcl::PointCloud<PointType>());
    c->reserve(n + 1);
    for (size_t i = 0; i < n; ++i) {
        PointType p;
        p.x = buf[3*i]; p.y = buf[3*i+1]; p.z = buf[3*i+2]; p.intensity = 1.f;
        c->push_back(p);
    }
    // One stray return far from the map. A single flyer is enough: pcl::VoxelGrid sizes its
    // dense index from the bounding box, not from the occupied volume.
    if (stray_m > 0.0) {
        PointType p;
        p.x = p.y = p.z = static_cast<float>(stray_m); p.intensity = 1.f;
        c->push_back(p);
    }
    c->width = c->size(); c->height = 1; c->is_dense = true;
    return c;
}

static void extent(const pcl::PointCloud<PointType> &c, double *span)
{
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (size_t i = 0; i < c.size(); ++i) {
        const float v[3] = {c[i].x, c[i].y, c[i].z};
        for (int k = 0; k < 3; ++k) { if (v[k] < mn[k]) mn[k] = v[k]; if (v[k] > mx[k]) mx[k] = v[k]; }
    }
    for (int k = 0; k < 3; ++k) span[k] = mx[k] - mn[k];
}

// The same arithmetic pcl::VoxelGrid::applyFilter does before deciding whether to bail out.
static double dense_cells(const double *span)
{
    double c = 1.0;
    for (int k = 0; k < 3; ++k) c *= static_cast<double>(static_cast<int64_t>(span[k] / LEAF) + 1);
    return c;
}

static size_t run_exact(const pcl::PointCloud<PointType>::Ptr &in)
{
    pcl::PointCloud<PointType>::Ptr out(new pcl::PointCloud<PointType>());
    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(LEAF, LEAF, LEAF);
    vg.setInputCloud(in);
    vg.filter(*out);
    return out->size();
}

static size_t run_octree(const pcl::PointCloud<PointType>::Ptr &in)
{
    pcl::octree::OctreePointCloudVoxelCentroid<PointType> oc(LEAF);
    oc.setInputCloud(in);
    oc.addPointsFromInputCloud();
    pcl::octree::OctreePointCloudVoxelCentroid<PointType>::AlignedPointTVector c;
    oc.getVoxelCentroids(c);
    return c.size();
}


static size_t run_approx(const pcl::PointCloud<PointType>::Ptr &in)
{
    pcl::PointCloud<PointType>::Ptr out(new pcl::PointCloud<PointType>());
    pcl::ApproximateVoxelGrid<PointType> avg;
    avg.setLeafSize(LEAF, LEAF, LEAF);
    avg.setInputCloud(in);
    avg.filter(*out);
    return out->size();
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <map_xyz.f32>\n", argv[0]); return 2; }

    // ------------------------------------------------- 1. ordinary map, index fits in int32
    printf("\n1. Real map, no stray return\n");
    {
        pcl::PointCloud<PointType>::Ptr in = load(argv[1], 0.0);
        double span[3]; extent(*in, span);
        printf("  %zu points, extent %.1f x %.1f x %.1f m -> %.3g dense cells at %.2f m "
               "(int32 max %.3g)\n", in->size(), span[0], span[1], span[2],
               dense_cells(span), LEAF, static_cast<double>(INT32_MAX));

        size_t ex = run_exact(in), ap = run_approx(in);
        double ratio = static_cast<double>(ap) / ex;
        printf("  pcl::VoxelGrid            -> %zu  (%.2fx reduction)\n", ex, (double)in->size() / ex);
        printf("  pcl::ApproximateVoxelGrid -> %zu  (%.2fx reduction, %.3f of exact)\n",
               ap, (double)in->size() / ap, ratio);
        check(ex < in->size() / 2, "exact voxeliser filters normally here");
        size_t oc = run_octree(in);
        printf("  OctreePointCloudVoxelCentroid -> %zu  (%.3f of exact)  <- shipped\n",
               oc, (double)oc / ex);
        check(ratio > 2.0,
              "ApproximateVoxelGrid leaves far more points than exact -- why it was replaced");
        check(oc > ex * 0.95 && oc < ex * 1.05,
              "octree agrees with the exact voxeliser within 5%");
    }

    // ------------------------------------------------- 2. one stray return, index overflows
    printf("\n2. Same map plus one stray return at 30 km\n");
    {
        pcl::PointCloud<PointType>::Ptr in = load(argv[1], 30000.0);
        double span[3]; extent(*in, span);
        printf("  %zu points, extent %.0f x %.0f x %.0f m -> %.3g dense cells at %.2f m\n",
               in->size(), span[0], span[1], span[2], dense_cells(span), LEAF);

        printf("  (expect a PCL warning on the next line)\n");
        size_t ex = run_exact(in);
        printf("  pcl::VoxelGrid            -> %zu\n", ex);
        check(ex == in->size(),
              "pcl::VoxelGrid returns the cloud UNFILTERED -- the defect being fixed");

        size_t ap = run_approx(in);
        printf("  pcl::ApproximateVoxelGrid -> %zu  (%.2fx reduction)\n",
               ap, (double)in->size() / ap);
        size_t oc = run_octree(in);
        printf("  OctreePointCloudVoxelCentroid -> %zu  (%.2fx reduction)  <- shipped\n",
               oc, (double)in->size() / oc);
        check(oc < in->size() / 5,
              "octree still downsamples with a huge bounding box -- no dense index to overflow");
    }

    // ------------------------------------------------- 3. the aliasing fix
    // The old code called filter(*globalCloud) with globalCloud also the input.
    printf("\n3. In-place filter(), input aliased to output\n");
    {
        pcl::PointCloud<PointType>::Ptr a = load(argv[1], 0.0);
        pcl::PointCloud<PointType>::Ptr sep(new pcl::PointCloud<PointType>());
        pcl::ApproximateVoxelGrid<PointType> a1;
        a1.setLeafSize(LEAF, LEAF, LEAF);
        a1.setInputCloud(a);
        a1.filter(*sep);

        pcl::PointCloud<PointType>::Ptr b = load(argv[1], 0.0);
        pcl::ApproximateVoxelGrid<PointType> a2;
        a2.setLeafSize(LEAF, LEAF, LEAF);
        a2.setInputCloud(b);
        a2.filter(*b);                       // the old pattern

        printf("  separate output %zu points\n", sep->size());
        printf("  aliased  output %zu points\n", b->size());
        check(sep->size() == b->size(),
              "same result either way on this PCL build -- aliasing is latent, not active");
    }

    printf("\n%s (%d failed)\n", failures ? "FAILURES" : "all checks passed", failures);
    return failures ? 1 : 0;
}
