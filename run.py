from argparse import ArgumentParser, Namespace


import os
from pathlib import Path
import itertools


parser = ArgumentParser(description="run 4K4DGen")
parser.add_argument('--conda_root',  type=str, default=None)
parser.add_argument('--animating_env', type=str, default=None)
parser.add_argument('--lifting_env', type=str, default=None)
parser.add_argument('--animating_folder', type=str, default='animating')
parser.add_argument('--lifting_folder', type=str, default='4dlifting')
parser.add_argument('--select_animating', action='store_true')
parser.add_argument('--animating_passes', type=int, default=5)
# parser.add_argument('--scene_path', type=str, default='example')
parser.add_argument('--input_img', type=str, default='data/example')
parser.add_argument('--input_mask', type=str, default=None)
parser.add_argument('--input_config', type=str, default=None)
parser.add_argument('--write_to', type=str, default='run.sh')
parser.add_argument('--write_to_folder', type=str, default='run_files')
parser.add_argument('--video_frames', type=int, default=14)
# parser.add_argument('--animating')
args = parser.parse_args()

animating_template = '''\
{} gen4k.py --config ./svd_mask_config.yaml --eval validation_data.prompt_image={} validation_data.mask={} validation_data.config={} pretrained_model_path=pretrained/animate_anything_svd_v1.0 --diff_mode {}\
'''

def generate_scene(
    scene_id,
    img_path, 
    mask_path, 
    config_path, 
    python_bin_animating,
    python_bin_lifting,
    video_frames,
):
    # do animating
    cmds = [
        f'echo \"now running {scene_id}\"',
        f'cd {args.animating_folder}'
    ]
    cmds.append(animating_template.format(python_bin_animating, img_path, mask_path, config_path, 'return_latents'))
    cmds.append(animating_template.format(python_bin_animating, img_path, mask_path, config_path, 'decode_latents'))
    if args.select_animating:
        cmds.extend(
            [
                'echo \"which video will you use?\"',
                'read videoid'
            ]
        )
    else:
        cmds.append('videoid=0')
    output_folder = os.path.join(args.animating_folder, 'output', img_path.split('.')[0])
    output_path = f'{output_folder}/${{videoid}}.gif'
    mask_path = f'{output_folder}/${{videoid}}_mask.png'
    # lifting_folder = os.path.join(f'{args.lifting_folder}', 'data', scene_id)
    # lifting_output = os.path.join(f'{args.lifting_folder}', 'output', scene_id)
    lifting_folder = os.path.join('data', scene_id)
    lifting_output = os.path.join('output', scene_id)
    
    cmds.append('cd ..')
    cmds.extend(
        [
            f'rm -rf {lifting_folder}',
            f'mkdir -p {lifting_folder}',
            f'cp {output_path} {lifting_folder}/{scene_id}.gif',
            f'cp {mask_path} {lifting_folder}/{scene_id}_mask.png'
        ]
    )

    cmds.append(f'cd {args.lifting_folder}')
    
    cmds.append(f'{python_bin_lifting} generate_init_geo_4k.py -s {lifting_folder} -m {lifting_output}')
    for k in range(video_frames):
        cmds.append(f'{python_bin_lifting} train.py -s {lifting_folder}/{k:02d} -m {lifting_output}/{k:02d}')
        
    #
    cmds.append('cd ..')
    # print('\n'.join(cmds))
    return cmds

def try_to_get_files(anime_root: Path, input_img, suffix):
   
    if not os.path.exists(input_img):
        _input_img = anime_root / input_img
        if not os.path.exists(_input_img):
            print(f'file {input_img} nor {_input_img} not found')
            return {}
        input_img = _input_img

        
    if os.path.isdir(input_img):
        img_files = os.listdir(input_img)
        img_files = [os.path.join(input_img, x) for x in img_files if x.lower().split('.')[-1] in suffix]
    elif input_img.lower().split('.')[-1] in suffix:
        img_files = [input_img]
    
    img_files = [x for x in img_files if x.lower().startswith(str(anime_root))]
    img_files = [x[len(str(anime_root)):].strip('/') for x in img_files]
 
    d_img_files = {
        x.split('/')[-1].split('.')[0]: x
        for x in img_files
    }
    return d_img_files

def get_scenes_list():
    anime_root = Path(args.animating_folder)
    input_img = args.input_img
    input_mask = args.input_mask
    input_config = args.input_config
    
    scene_imgs = try_to_get_files(anime_root, input_img, suffix=['jpg', 'png', 'jpeg'])
    if len(scene_imgs) == 0:
        raise Exception('No input image found')
    
    if input_mask is None:
        input_mask = '/'.join(list(scene_imgs.values())[0].split('/')[:-1]) + '_mask'
        
    if input_config is None:
        input_config = '/'.join(list(scene_imgs.values())[0].split('/')[:-1]) + '_config'
        
        
    scene_masks = try_to_get_files(anime_root, input_mask, suffix=['jpg', 'png', 'jpeg'])
    scene_configs = try_to_get_files(anime_root, input_config, suffix=['json'])
    rtn = {}
    for k in scene_imgs.keys():
        assert k in scene_masks
        assert k in scene_configs
        rtn[k] = (
            scene_imgs[k],
            scene_masks[k],
            scene_configs[k]
        )
    return rtn
        
    
    
def run():
    print('searching running environment')
    conda_root = Path(args.conda_root) if args.conda_root else Path.home() / 'anaconda3'
    _lifting_env = args.lifting_env if args.lifting_env else '4dlifting'
    _animating_env = args.animating_env if args.animating_env else 'animating'
    lifting_env = get_python_env(conda_root, _lifting_env)
    animating_env = get_python_env(conda_root, _animating_env)
    
    print('searching scenes to be run')
    to_run_list = get_scenes_list()
    
    print('generating running scripts')
    run_files = []
    write_to_folder = args.write_to_folder
    write_to = args.write_to
    os.makedirs(write_to_folder, exist_ok=True)
    
    video_frames = args.video_frames
    
    
    for scene_name, files in to_run_list.items():
        img_p, mask_p, cfg_p = files
        cmds = generate_scene(
            scene_name, 
            img_p, 
            mask_p, 
            cfg_p, 
            animating_env,
            lifting_env, 
            video_frames
        )
        write_to_file = os.path.join(write_to_folder, f'{scene_name}.sh')
        with open(write_to_file, 'w') as fout:
            fout.write('\n'.join(cmds))
        run_files.append('bash ' + write_to_file)
        print(f'running script for scene {scene_name} write to {write_to_file}')
    
    with open(write_to, 'w') as fout:
        fout.write('\n'.join(run_files))
    print(f'running script write to {write_to}, to run 4k4dgen, simply run: \nbash {write_to}')
    
        
    # get the scene list
    # if 
    
def get_python_env(conda_root:Path, env):
    try:
        user_paths = os.environ['PATH'].split(os.pathsep)
    except KeyError:
        user_paths = []
    
    
    env = Path(env)
    possible_list = [
        env, 
        env / 'bin' / 'python',
        conda_root / 'envs' / env, 
        conda_root / 'envs' / env / 'bin' / 'python',
        'python',
    ]
    
    path_search = list(itertools.product(user_paths, possible_list))
    path_search = [os.path.join(x[0], x[1]) for x in path_search]
    possible_list.extend(path_search)
    
    
    for possible_env in possible_list:
        if os.path.exists(possible_env) and str(possible_env).endswith('python'):
            return possible_env
   
    raise Exception('''none of the env exists: ({})'''.format(', '.join(possible_list)))
    
        
    
if __name__ == '__main__':
    run()
    # generate_scene(
    #     'I2', 
    #     'data/example/I2.jpg', 
    #     'data/example_mask/I2.png', 
    #     'data/example_config/I2.json', 
    #     'python', 
    #     'python',
    # )