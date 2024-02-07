base_url = '/Users/tobias/data/5Safe/vup/homography_evaluation'
data = {
    'DJI_0026': {
        'ref_pts': ['w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w19', 'w21', 'w22'],
        'val_pts': ['b6', 'b7', 'b8', 'b9', 'b19', 'b21'] 
    },
    'DJI_0029': {
        'ref_pts': ['w9', 'w22', 'w19', 'w23', 'w24', 'w20', 'w25', 'w26', 'w28', 'w21'],
        'val_pts': ['b19', 'b21', 'b22', 'b23', 'b20', 'b28', 'b27', 'b32', 'b34', 'b33', 'b24']
    },
    'DJI_0032': {
        'ref_pts': ['w17', 'w16', 'w14', 'w13', 'w11', 'w12', 'w10', 'w15', 'w1', 'w2'],
        'val_pts': ['b16', 'b15', 'b14', 'b12', 'b11', 'b10', 'b1', 'b2', 'b13']
    },
    'DJI_0035': {
        'ref_pts': ['w19', 'w21', 'w25', 'w20', 'w18', 'w7', 'w5', 'w4', 'w6', 'w3', 'w1', 'w2', 'w15', 'w13', 'w14', 'w17'],
        'val_pts': ['b28', 'b27', 'b20', 'b18', 'b21', 'b19', 'b7', 'b6', 'b5', 'b3', 'b4', 'b13', 'b11', 'b12', 'b14', 'b15', 'b16', 'b17']
    },
    'DJI_0038': {
        'ref_pts': ['w20', 'w19', 'w7', 'w5', 'w6', 'w4', 'w3', 'w18', 'w17', 'w16', 'w13', 'w14', 'w11', 'w10'],
        'val_pts': ['b20', 'b18', 'b17', 'b16', 'b15', 'b14', 'b13', 'b12', 'b11', 'b10', 'b3', 'b4', 'b5', 'b6']
    },
    'DJI_0045': {
        'ref_pts': ['w11', 'w12', 'w13', 'w14', 'w15'],
        'val_pts': ['b10', 'b12', 'b25']
    },
    'DJI_0049': {
        'ref_pts': ['w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w18', 'w20', 'w5', 'w6', 'w7', 'w3', 'w22', 'w23', 'w24'],
        'val_pts': ['b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b4', 'b5', 'b6', 'b8', 'b9']
    },
    'DJI_0053': {
        'ref_pts': ['w1', 'w2', 'w3', 'w12', 'w14', 'w15', 'w16', 'w17'],
        'val_pts': ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b12', 'b13', 'b14', 'b15', 'b16']
    },
    'DJI_0061': {
        'ref_pts': ['w2', 'w5', 'w8', 'w13', 'w14', 'w15', 'w16', 'w17', 'w19', 'w25', 'w26'],
        'val_pts': ['b3', 'b4', 'b13', 'b14', 'b15', 'b16', 'b17', 'b27', 'b28', 'b29']
    },
    'DJI_0066': {
        'ref_pts': ['w6', 'w7', 'w19', 'w20', 'w21', 'w22', 'w23', 'w24', 'w25', 'w26', 'w27'],
        'val_pts': ['b18', 'b19', 'b21', 'b22', 'b20', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34']
    },
    'DJI_0067': {
        'ref_pts': ['w6', 'w8', 'w9', 'w21', 'w22', 'w23'],
        'val_pts': ['b8', 'b9', 'b19', 'b21', 'b22', 'b23']
    },
    'DJI_0078': {
        'ref_pts': ['w2', 'w4', 'w5', 'w6', 'w7', 'w10', 'w11', 'w12', 'w13', 'w14', 'w16', 'w17', 'w18', 'w19', 'w20', 'w21', 'w25'],
        'val_pts': ['b4', 'b6', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b27', 'b28', 'b29']
    }
}


for img_name, pts in data.items():
    ref_pts = pts['ref_pts']
    val_pts = pts['val_pts']

    config = f'''perspective_view:
  fname_img: '{base_url}/data/perspective_views/{img_name}.JPG'
  fname_val_pts: '{base_url}/annotations/{img_name}_blue.txt'
  fname_ref_pts: '{base_url}/annotations/{img_name}_white.txt'
  fname_fov_pts: '{base_url}/annotations/{img_name}_fov_pts.txt'
  fname_K: '{base_url}/data/calibration/mini3.json'
top_view:
  fname_img: '{base_url}/data/top_view/DJI_0017.JPG'
  fname_val_pts: '{base_url}/annotations/DJI_0017_blue.txt'
  fname_ref_pts: '{base_url}/annotations/DJI_0017_white.txt'
  scaling_factor: 54.67
selection:
  ref_pts: {ref_pts}
  val_pts: {val_pts}
output:
  dir: 'out/{img_name}.JPG'
'''
    with open(f'conf/config_{img_name}.yaml', 'w') as file:
      file.write(config) 

    