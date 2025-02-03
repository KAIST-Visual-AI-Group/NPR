
from typing import Literal, Optional

from jaxtyping import Shaped, jaxtyped
import torch
import torch.nn as nn
from typeguard import typechecked

from .encoder.pointransformer import PointTransformerEncoder
from .decoder.crosstransformer_decoder import CrossTransformerDecoder
from .utils import compute_l2_error


class Deformation_Networks(nn.Module):

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        data_type: Literal["vertices", "jacobians"] = "jacobians",
    ):
        super(Deformation_Networks, self).__init__()

        self.data_type = data_type
        if self.data_type == "vertices":
            has_features = False
            inp_feat_dim = 0
            dec_out_dim = 3
        elif self.data_type == "jacobians":
            has_features = True
            inp_feat_dim = 9  # Per-triangle Jacobian
            dec_out_dim = 9
        else:
            raise ValueError(
                f"Unsupported data type {str(self.data_type)}"
            )

        encoder_kwargs = {
            'npoints_per_layer': [5000, 500, 100],
            'nneighbor': 16,
            'nneighbor_reduced': 10,
            'nfinal_transformers': 3,
            'd_transformer': 256,
            'd_reduced': 120,
            'full_SA': True,
        }
        self.encoder = PointTransformerEncoder(
            has_features=has_features, inp_feat_dim=inp_feat_dim,
            **encoder_kwargs,
        )
            
        decoder_kwargs = {
            'dim_inp': 256,
            'dim': 200,
            'nneigh': 7,
            'hidden_dim': 128,
            'out_dim': dec_out_dim,
        }
        self.decoder = CrossTransformerDecoder(
            **decoder_kwargs
        )

    @jaxtyped(typechecker=typechecked)
    def encode(self, x):
        """
        x: (B x V x 3) or (B x F x 12)
        """

        B = x.shape[0]
        D = x.shape[2]
        
        if self.data_type == "vertices":
            assert D == 3, f"Expected 3, got {D}"
        elif self.data_type == "jacobians":
            assert D == 12, f"Expected 12, got {D}"
        else:
            raise ValueError(f"Unsupported data type {str(self.data_type)}")

        return self.encoder(x)

    @jaxtyped(typechecker=typechecked)
    def decode(self, encoding, y):
        """
        encoding: outputs from the function 'encode'
        y: (B x V x 3): the points to query the encodings
        """
        return self.decoder(y, encoding)

    @jaxtyped(typechecker=typechecked)
    def get_z_from_anchor_and_feat(
        self,
        anchors: Shaped[torch.Tensor, "B N 3"],
        anchor_feats: Shaped[torch.Tensor, "B N 256"],
    ) -> Shaped[torch.Tensor, "B 256"]:
        """
        anchors: (B x N x 3)
        anchor_feats: (B x N x 256)
        """
        lat_vec = anchor_feats.max(dim=1)[0]
        z = self.encoder.fc_middle(lat_vec)
        return z

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        points: Shaped[torch.Tensor, "B F *"],
        # surface_samples_inputs: Shaped[torch.Tensor, "B V *"],
        surface_samples_inputs,
        beta: Shaped[torch.Tensor, "B"],
        centroids: Optional[Shaped[torch.Tensor, "B F 3"]] = None,
    ):
        """
        points: B x N x D
        surface_sample_inputs: 
        """
        batch_size = points.shape[0]
        num_points = points.shape[1]
        
        ################################################################################################
        # Geometry & Flow encoding.
        ################################################################################################     
        if self.J_reg:  # NOTE: 20240503 Debugging Jacobian regression
            
            # print(f"encoder in: {surface_samples_inputs.shape}")

            encoding = self.encoder(surface_samples_inputs)
        
        else:
            if self.no_input_corr:
                ####
                # encoding = self.encoder(surface_samples_inputs[:, :, 0:3].contiguous())
                encoding = self.encoder(surface_samples_inputs)
                ####
            else:
                ####
                # encoding = self.encoder(surface_samples_inputs)
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                enc_in = torch.cat([centroids, points], dim=-1)

                # print(f"enc_in shape: {enc_in.shape}")
                encoding = self.encoder(enc_in)

                # print(encoding["z"].shape)
                # print(encoding["anchors"].shape)
                # print(encoding["anchor_feats"].shape)
                ####
        ################################################################################################
        # Flow decoding.
        ################################################################################################
        ####
        # deformed_points = self.decoder(points, encoding)        
        deformed_points = self.decoder(
            points,
            encoding,
            beta,
            centroids,
        )
        ####

        return deformed_points
    
    
def train_on_batch_with_cano(model, optimizer, data_dict, config):
    # Make sure that everything has the correct size
    optimizer.zero_grad()
    surface_samples_inputs = data_dict['surface_samples_inputs']
    source_points = data_dict['space_samples_src']
    target_points = data_dict['space_samples_tgt']
    deformed_points = model(source_points, surface_samples_inputs)
    # Compute the loss
    loss = compute_l2_error(deformed_points, target_points)
    # Do the backpropagation
    loss.backward()
    # Do the update
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate_on_batch_with_cano(model, data_dict, config):
    surface_samples_inputs = data_dict['surface_samples_inputs']
    source_points = data_dict['space_samples_src']
    target_points = data_dict['space_samples_tgt']
    deformed_points = model(source_points, surface_samples_inputs)
    # Compute the loss
    loss = compute_l2_error(deformed_points, target_points)
    return loss.item()

@torch.no_grad()
def test_on_batch_with_cano(model, data_dict, config, compute_loss=False):
    surface_samples_inputs = data_dict['surface_samples_inputs']
    source_points = data_dict['surface_samples_src']
    target_points = data_dict['surface_samples_tgt']
    
    deformed_points = model(source_points, surface_samples_inputs)
    data_dict['surface_samples_tgt_pred'] = deformed_points
    
    source_verts = data_dict['verts_src']
    target_verts = data_dict['verts_tgt']
    deformed_verts = model(source_verts, surface_samples_inputs)
    data_dict['verts_tgt_pred'] = deformed_verts
    
    # Compute the loss
    if compute_loss:
        loss = compute_l2_error(deformed_verts, target_verts)
    else:
        loss = torch.zeros((1), dtype=torch.float32)
    return loss.item(), data_dict


if __name__ == "__main__":

    from pathlib import Path
    
    import pickle

    device = torch.device("cuda")

    cfg_file = Path("forward_net_cfg.pkl")
    with open(cfg_file, "rb") as f:
        cfg = pickle.load(f)
    net = Deformation_Networks(cfg, True).to(device)
    print("got network")

    pts_dummy = torch.randn((3, 25000, 9)).to(device)
    surf_pts_dummy = torch.randn((3, 5000, 3)).to(device)
    f_centroids = torch.randn((3, 25000, 3)).to(device)
    beta_dummy = torch.randn((3, )).to(device)
    _ = net(pts_dummy, surf_pts_dummy, beta_dummy, f_centroids)
    print("done")
