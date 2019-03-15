# This file is part of NFDMLab.
#
# NFDMLab is free software; you can redistribute it and/or
# modify it under the terms of the version 2 of the GNU General
# Public License as published by the Free Software Foundation.
#
# NFDMLab is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with NFDMLab; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# 02111-1307 USA
#
# Contributors:
# Marius Brehler (TU Dortmund) 2018-2019

import ipywidgets as widgets
from ipywidgets import Layout
import numpy as np

style = {'description_width': 'initial'}

def select_qam_level(default_value):
    w = widgets.Select(
        options=[4,16,64],
        value=default_value,
        # rows=10,
        description='QAM Level',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style
    )
    return w

def select_constellation(available_constellations,default_value):
    w = widgets.Select(
        options=available_constellations.keys(),
        value=default_value,
        # rows=10,
        description='',
        disabled=False,
        continuous_update=False,
        style=style,
        layout=widgets.Layout(width='25%')
        #layout=widgets.Layout(width='20%')
    )
    return w

def select_n_blocks(available_n_blocks,default_value):
    w = widgets.Select(
        options=available_n_blocks,
        value=default_value,
        # rows=10,
        description='',
        disabled=False,
        style=style,
        layout=widgets.Layout(width='25%')
    )
    return w

def select_mfactor(default_value):
    w = widgets.Select(
        options=[1],
        value=default_value,
        # rows=10,
        description='',
        disabled=False,
        style=style,
        layout=widgets.Layout(width='25%')
    )
    return w

def select_carrier_waveform(available_carrier_waveforms, default_value):
    w = widgets.Select(
        options=sorted(available_carrier_waveforms.keys()),
        value=default_value,
        # rows=10,
        description='',
        disabled=False,
        style=style,
        layout=widgets.Layout(width='25%')
        #layout=widgets.Layout(width='35%')
    )
    return w

def select_alpha(available_alphas, default_value):
    w = widgets.Select(
        options=sorted(available_alphas.keys()),
        value=default_value,
        # rows=10,
        description='',
        disabled=False,
        style=style,
        layout=widgets.Layout(width='25%')
    )
    return w

def select_amplification(available_amplification, default_value):
    w = widgets.Select(
        options=available_amplification,
        value=default_value,
        # rows=10,
        description='',
        disabled=False,
        style=style,
        layout=widgets.Layout(width='25%')
    )
    return w

def select_noise():
    w = widgets.Checkbox(
        value=False,
        description='Add noise',
        disabled=False
    )
    return w

def select_power_normalization(available_power_normalization, default_value):
    w = widgets.Select(
        options=sorted(available_power_normalization.keys()),
        value=default_value,
        # rows=10,
        description='',
        disabled=False,
        style=style,
        layout=widgets.Layout(width='25%')
        #layout=widgets.Layout(width='32%')
    )
    return w

def select_path_average():
    w = widgets.Checkbox(
        value=False,
        description='Path average',
        disabled=False
    )
    return w


def update_alpha(choosen_amplification,noise_widget,path_average_widget):
    if choosen_amplification == 'Lossless':
        selected_alpha = np.array([0.0])
        noise_widget.disabled = True
        noise_widget.value = False
        path_average_widget.disabled = True
        path_average_widget.value = False
    elif choosen_amplification == 'EDFA':
        selected_alpha = np.array([0.2e-3])
        noise_widget.disabled = False
        noise_widget.value = False
        path_average_widget.disabled = False
        path_average_widget.value = True
    elif choosen_amplification == "Raman":
        from scipy.io import loadmat
        matfile = loadmat('../data/gainprofile_40steps.mat')
        selected_alpha = matfile['gain'].flatten()
        noise_widget.disabled = True
        noise_widget.value = False
        path_average_widget.disabled = True
        path_average_widget.value = False
    else:
        raise Exception('alpha/gain method not supported')

    return selected_alpha


def create_hbox_modulationHeader():
    table_header_1_widget = widgets.Button( description='Constellation',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   )
    table_header_2_widget = widgets.Button( description='Carrier waveform',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   ) 
    table_header_3_widget = widgets.Button( description='Power normalization',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   )

    hbox_modH = widgets.HBox([table_header_1_widget, table_header_2_widget, table_header_3_widget])

    return hbox_modH

def create_hbox_modulationHeader22():
    table_header_1_widget = widgets.Button( description='Constellation',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   )
    table_header_2_widget = widgets.Button( description='Multiplier Factor',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   ) 
    table_header_3_widget = widgets.Button( description='Power normalization',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   )

    hbox_modH = widgets.HBox([table_header_1_widget, table_header_2_widget, table_header_3_widget])

    return hbox_modH

def create_hbox_modulationHeader2():
    table_header_1_widget = widgets.Button( description='Constellation',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   )


    hbox_modH = widgets.HBox([table_header_1_widget])

    return hbox_modH

def create_hbox_modulationHeader3():
    table_header_1_widget = widgets.Button( description='Constellation',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   )
    table_header_3_widget = widgets.Button( description='Power normalization',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   )

    hbox_modH = widgets.HBox([table_header_1_widget, table_header_3_widget])

    return hbox_modH

def create_hbox_transmitHeader():
    table_header_4_widget = widgets.Button( description='Number of blocks',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   )

    hbox_transmitH = widgets.HBox([table_header_4_widget])

    return hbox_transmitH

def create_hbox_linkHeader():
    table_header_5_widget = widgets.Button( description='Amplification scheme',
                                    disabled=True,
                                    button_style='',
                                    tooltip='',
                                    icon='',
                                    layout=widgets.Layout(width='25%')
                                   )

    hbox_linkH = widgets.HBox([table_header_5_widget])

    return hbox_linkH