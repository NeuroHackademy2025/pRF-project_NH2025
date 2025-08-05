#  Imports
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

def plotly_template(template_specs):
    """
    Define the template for plotly
    Parameters
    ----------
    template_specs : dict
        dictionary contain specific figure settings
    
    Returns
    -------
    fig_template : plotly.graph_objs.layout._template.Template
        Template for plotly figure
    """
    import plotly.graph_objects as go
    fig_template=go.layout.Template()

    # Violin plots
    fig_template.data.violin = [go.Violin(
                                    box_visible=False,
                                    points=False,
                                    # opacity=1,
                                    line_color= "rgba(0, 0, 0, 1)",
                                    line_width=template_specs['plot_width'],
                                    width=0.8,
                                    #marker_symbol='x',
                                    #marker_opacity=1,
                                    hoveron='violins',
                                    meanline_visible=False,
                                    # meanline_color="rgba(0, 0, 0, 1)",
                                    # meanline_width=template_specs['plot_width'],
                                    showlegend=False,
                                    )]

    # Barpolar
    fig_template.data.barpolar = [go.Barpolar(
                                    marker_line_color="rgba(0,0,0,1)",
                                    marker_line_width=template_specs['plot_width'], 
                                    showlegend=False, 
                                    )]
    # Pie plots
    fig_template.data.pie = [go.Pie(textposition=["inside","none"],
                                    # marker_line_color=['rgba(0,0,0,1)','rgba(255,255,255,0)'],
                                    marker_line_width=0,#[template_specs['plot_width'],0],
                                    rotation=0,
                                    direction="clockwise",
                                    hole=0.4,
                                    sort=False,
                                    )]

    # Layout
    fig_template.layout = (go.Layout(# general
                                    font_family=template_specs['font'],
                                    font_size=template_specs['axes_font_size'],
                                    plot_bgcolor=template_specs['bg_col'],

                                    # # x axis
                                    xaxis_visible=True,
                                    xaxis_linewidth=template_specs['axes_width'],
                                    xaxis_color= template_specs['axes_color'],
                                    xaxis_showgrid=False,
                                    xaxis_ticks="outside",
                                    xaxis_ticklen=8,
                                    xaxis_tickwidth = template_specs['axes_width'],
                                    xaxis_title_font_family=template_specs['font'],
                                    xaxis_title_font_size=template_specs['title_font_size'],
                                    xaxis_tickfont_family=template_specs['font'],
                                    xaxis_tickfont_size=template_specs['axes_font_size'],
                                    xaxis_zeroline=False,
                                    xaxis_zerolinecolor=template_specs['axes_color'],
                                    xaxis_zerolinewidth=template_specs['axes_width'],
                                    # xaxis_range=[0,1],
                                    xaxis_hoverformat = '.1f',
                                    
                                    # y axis
                                    yaxis_visible=True,
                                    yaxis_linewidth=template_specs['axes_width'],
                                    yaxis_color= template_specs['axes_color'],
                                    yaxis_showgrid=False,
                                    yaxis_ticks="outside",
                                    yaxis_ticklen=8,
                                    yaxis_tickwidth = template_specs['axes_width'],
                                    yaxis_tickfont_family=template_specs['font'],
                                    yaxis_tickfont_size=template_specs['axes_font_size'],
                                    yaxis_title_font_family=template_specs['font'],
                                    yaxis_title_font_size=template_specs['title_font_size'],
                                    yaxis_zeroline=False,
                                    yaxis_zerolinecolor=template_specs['axes_color'],
                                    yaxis_zerolinewidth=template_specs['axes_width'],
                                    yaxis_hoverformat = '.1f',

                                    # bar polar
                                    polar_radialaxis_visible = False,
                                    polar_radialaxis_showticklabels=False,
                                    polar_radialaxis_ticks='',
                                    polar_angularaxis_visible = False,
                                    polar_angularaxis_showticklabels = False,
                                    polar_angularaxis_ticks = ''
                                    ))

    # Annotations
    fig_template.layout.annotationdefaults = go.layout.Annotation(
                                    font_color=template_specs['axes_color'],
                                    font_family=template_specs['font'],
                                    font_size=template_specs['title_font_size'])

    return fig_template

def prf_ecc_pcm_plot(df_ecc_pcm, data_pcm_raw, fig_width, fig_height, rois, roi_colors, plot_groups, max_ecc):
    """
    Make scatter plot for relationship between eccentricity and pCM

    Parameters
    ----------
    df_ecc_pcm : dataframe for the plot
    fig_width : figure width in pixels
    fig_height : figure height in pixels
    rois : list of rois
    roi_colors : dictionary with keys as roi and value correspondig rgb color
    plot_groups : groups of roi to plot together
    max_ecc : maximum eccentricity
    
    Returns
    -------
    fig : eccentricy as a function of pcm plot
    """
    from math_utils import weighted_regression, weighted_nan_percentile, weighted_nan_median
    from sklearn.metrics import r2_score
    
    # General figure settings
    template_specs = dict(axes_color="rgba(0, 0, 0, 1)",
                          axes_width=2,
                          axes_font_size=15,
                          bg_col="rgba(255, 255, 255, 1)",
                          font='Arial',
                          title_font_size=15,
                          plot_width=1.5)
    
    # General figure settings
    fig_template = plotly_template(template_specs)

    # General settings
    rows, cols = 1, len(plot_groups)
    fig = make_subplots(rows=rows, cols=cols, print_grid=False)
    
    for l, line_label in enumerate(plot_groups):
        for j, roi in enumerate(line_label):

            # Parametring colors
            roi_color = roi_colors[roi]
            roi_color_opac = f"rgba{roi_color[3:-1]}, 0.15)"
            
            # Get data
            df = df_ecc_pcm.loc[(df_ecc_pcm.roi == roi)]
            data_pcm_raw_roi = data_pcm_raw.loc[(data_pcm_raw.roi == roi)]
            
            ecc_median = np.array(df.prf_ecc)
            pcm_median = np.array(df.pRF_CM)
            r2_median = np.array(df.prf_r2)
            # pcm_upper_bound = np.array(df.prf_pcm_bins_ci_upper_bound)
            # pcm_lower_bound = np.array(df.prf_pcm_bins_ci_lower_bound)
            
            # Linear regression
            slope, intercept = weighted_regression(ecc_median, pcm_median, r2_median, model='pcm')
            
            # slope_upper, intercept_upper = weighted_regression(ecc_median[~np.isnan(pcm_upper_bound)], 
            #                                                    pcm_upper_bound[~np.isnan(pcm_upper_bound)], 
            #                                                    r2_median[~np.isnan(pcm_upper_bound)], 
            #                                                    model='pcm')
            
            # slope_lower, intercept_lower = weighted_regression(ecc_median[~np.isnan(pcm_lower_bound)], 
            #                                                    pcm_lower_bound[~np.isnan(pcm_lower_bound)], 
            #                                                    r2_median[~np.isnan(pcm_lower_bound)], 
            #                                                    model='pcm')

            line_x = data_pcm_raw_roi['prf_ecc']
            line = 1 / (slope * line_x + intercept)
            # line_upper = 1 / (slope_upper * line_x + intercept_upper)
            # line_lower = 1 / (slope_lower * line_x + intercept_lower)

            horton_line = 17.3/(data_pcm_raw_roi['prf_ecc'] + 0.75)
            
            # Filtrer nan
            valid_idx = ~np.isnan(pcm_median) & ~np.isnan(ecc_median)
            x_valid = ecc_median[valid_idx]
            y_valid = pcm_median[valid_idx]
            
            # make prediction
            y_pred = 1 / (slope * x_valid + intercept)
            
            # compute r2 
            r2 = r2_score(y_valid, y_pred)
            # Markers all point
            fig.add_trace(go.Scatter(x=data_pcm_raw_roi['prf_ecc'], 
                                     y=data_pcm_raw_roi['pRF_CM'], 
                                     mode='markers', 
                                     marker=dict(color=roi_color, 
                                                 symbol='circle', 
                                                 size=6, 
                                                 line=dict(color=roi_color, width=3)), 
                                     cliponaxis=False,
                                     showlegend=False, 
                                     opacity=0.4), 
                          row=1, col=l + 1)


            fig.add_trace(go.Scatter(x=line_x, 
                                     y=line, 
                                     mode='markers', 
                                     name=roi, 
                                     legendgroup=roi, 
                                     line=dict(color=roi_color, width=3), 
                                     showlegend=False), 
                          row=1, col=l+1)

            # Horton prediction
            fig.add_trace(go.Scatter(x=data_pcm_raw_roi['prf_ecc'], 
                         y=horton_line, 
                         mode='markers', 
                         name=roi, 
                         legendgroup=roi, 
                         line=dict(color='black', width=3), 
                         showlegend=False), 
              row=1, col=l+1)

            # # Error area
            # fig.add_trace(go.Scatter(x=np.concatenate([line_x, line_x[::-1]]),
            #                           y=np.concatenate([list(line_upper), list(line_lower[::-1])]), 
            #                           mode='lines', fill='toself', fillcolor=roi_color_opac, 
            #                           line=dict(color=roi_color_opac, width=0), showlegend=False), 
            #               row=1, col=l+1)


            # # Bin markers
            # fig.add_trace(go.Scatter(x=ecc_median, 
            #                          y=pcm_median, 
            #                          mode='markers', 
            #                          error_y=dict(type='data', 
            #                                       array=pcm_upper_bound - pcm_median, 
            #                                       arrayminus=pcm_median - pcm_lower_bound,
            #                                       visible=True, 
            #                                       thickness=3, 
            #                                       width=0, 
            #                                       color=roi_color),
            #                          marker=dict(color=roi_color, 
            #                                      symbol='square',
            #                                      size=8, line=dict(color=roi_color,
            #                                                        width=3)), 
            #                          showlegend=False), 
            #               row=1, col=l + 1)
            
            # Add legend
            annotation = go.layout.Annotation(x=5, y=20, text='{}, R<sup>2</sup>={:.2f}'.format(roi,r2), xanchor='left',
                                              showarrow=False, font_color=roi_color, 
                                              font_family=template_specs['font'],
                                              font_size=template_specs['axes_font_size'],
                                             )
            fig.add_annotation(annotation, row=1, col=l+1)

        # Set axis titles only for the left-most column and bottom-most row
        fig.update_yaxes(title_text='pRF cortical magn. (mm/dva)', row=1, col=1)
        fig.update_xaxes(title_text='pRF eccentricity (dva)', range=[0, max_ecc], showline=True, row=1, col=l+1)
        fig.update_yaxes(range=[0, 20], showline=True)
        # fig.update_layout(height=fig_height, width=fig_width, showlegend=False, template=fig_template,
        #                  # margin_l=100, margin_r=50, margin_t=50, margin_b=100
        #                  )
        fig.update_layout(showlegend=False, template=fig_template)
        
    return fig