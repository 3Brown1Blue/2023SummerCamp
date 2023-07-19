import open3d as o3d

if __name__=="__main__":
    ply_path='./results/phototourism/{epoch:d}_epoch=2-step=1042827/mesh/extracted_mesh_level_10_colored.ply'

    pcd=o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])