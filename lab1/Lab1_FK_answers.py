import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    Temp_stack = []
    cnt = 0
    with open(bvh_file_path,'r') as bvh_obj:
        for line in bvh_obj:
            Linelist = line.split()

            if Linelist[0] == 'ROOT':
                joint_name.append(Linelist[1])
                joint_parent.append(-1)
            elif Linelist[0] == 'JOINT':
                joint_name.append(Linelist[1])
                joint_parent.append(Temp_stack[-1])
            elif Linelist[0] == 'End':
                joint_name.append(joint_name[-1]+'_end')
                joint_parent.append(Temp_stack[-1])
            elif Linelist[0] == 'OFFSET':
                joint_offset.append([float(Linelist[1]), float(Linelist[2]), float(Linelist[3])])  
            elif Linelist[0] == '{':
                Temp_stack.append(cnt)
                cnt += 1 
            elif Linelist[0] == '}':
                Temp_stack.pop()            

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """

    # 默认Channel设置是RootJoint节点有6个channel(平移和旋转)，其余子节点有3个，末端节点没有channel。
    # 所以motion data前三个数字存放的是根节点位置，之后的数据才是存放的旋转数据
    Root_position = np.zeros(3, dtype=np.float64)
    for i in range(3):
        Root_position[i] = motion_data[frame_id][i]

    Channel_limit = int(len(motion_data[frame_id])/3) - 1
    NoEndJoint_EulerRotations = np.zeros((Channel_limit,3), dtype=np.float64)
    for i in range(Channel_limit):
        for j in range(3):
            NoEndJoint_EulerRotations[i][j] = motion_data[frame_id][3*(i+1)+j]

    cnt = 0
    Joint_num = len(joint_name)
    joint_positions = np.zeros((Joint_num, 3), dtype=np.float64)
    joint_orientations = np.zeros((Joint_num, 4), dtype=np.float64)

    for i in range(Joint_num):
        if joint_parent[i] == -1:
            joint_positions[i] = Root_position
            joint_orientations[i] = R.from_euler('XYZ', NoEndJoint_EulerRotations[cnt], degrees=True).as_quat()
        else:
            if '_end' not in joint_name[i]:
                cnt += 1
            joint_orientations[i] = (R.from_quat(joint_orientations[joint_parent[i]]) * R.from_euler('XYZ', NoEndJoint_EulerRotations[cnt], degrees=True)).as_quat()
            joint_positions[i] = joint_positions[joint_parent[i]] + R.from_quat(joint_orientations[joint_parent[i]]).apply(joint_offset[i])


    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    APose_joint_name, APose_joint_parent, APose_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    TPose_joint_name, TPose_joint_parent, TPose_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    APose_motion_data = load_motion_data(A_pose_bvh_path)


    Joint_num = len(APose_joint_name)

    motion_data = np.zeros(APose_motion_data.shape, dtype=np.float64)
    Frame_num, Channel_num = motion_data.shape

    for i in range(Frame_num):
        for j in range(3):
            motion_data[i][j] = APose_motion_data[i][j]

        cntA = 0
        for j in range(Joint_num):
            if '_end' not in TPose_joint_name[j]:
                cntT = 0
                for k in range(Joint_num):
                    if '_end' not in APose_joint_name[k]:
                        if TPose_joint_name[j] == APose_joint_name[k]:
                            for t in range(3):
                                motion_data[i][3*(cntA+1)+t] = APose_motion_data[i][3*(cntT+1)+t]
                            if TPose_joint_name[j] == 'lShoulder':
                                motion_data[i][3*(cntA+1)+2] -= 45.
                            elif TPose_joint_name[j] == 'rShoulder':
                                motion_data[i][3*(cntA+1)+2] += 45.
                        cntT += 1
                cntA += 1

    return motion_data
