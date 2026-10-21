"""Rebuild the seven manuscript figures from explicit local inputs.

No observations are distributed by this module. Inputs are checked before any
scientific dependencies are imported or an output directory is created. The
published figure functions retain the scientific calculations and layout of
the completed fivefold manuscript workflow. See FIGURES.md for input scope.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys

BLUE='#287C9E';ORANGE='#CE803A';TEAL='#25887F';GRAY='#83909B';INK='#243746';LIGHT='#DCE4E8';PURPLE='#7865A7'

def save(f,n,data=None):
 f.canvas.draw()
 map_axes=[a for a in f.axes if getattr(a,'_national_map',False)]
 for j,a in enumerate(map_axes):
  coords=map_data()['coords']
  pix=a.transData.transform(np.column_stack([coords._x,coords._y]))
  bb=a._south_inset.get_window_extent()
  occluded=(pix[:,0]>=bb.x0)&(pix[:,0]<=bb.x1)&(pix[:,1]>=bb.y0)&(pix[:,1]<=bb.y1)
  assert not occluded.any(), 'South China Sea inset covers a gauge'
  record=MAP_QA[-len(map_axes)+j]
  record.update(figure=n,panel=str(n)+('ad'[j] if n==7 else 'a'),
      inset_obscured_stations=int(occluded.sum()),geometry_coverage=map_data()['geometry_coverage'])
 for ext in ['png','svg','pdf']:f.savefig(FIG/f'Figure_{n}.{ext}',dpi=400)
 if data is not None:pd.DataFrame(data).to_csv(FIG/f'Figure_{n}_plot_data.csv',index=False,encoding='utf-8-sig')
 plt.close(f);print('SAVED',n,flush=True)

def panel(a,k,t):
 a.set_title(f'{k}  {t}',loc='left',pad=11,fontweight='bold')
 a.grid(axis='y',color=LIGHT,lw=.55);a.set_axisbelow(True)

def map_title(f,spec,k,t):
 p=spec.get_position(f);f.text(p.x0,p.y1+.022,f'{k}  {t}',fontsize=9,fontweight='bold',va='bottom')

def row(method,domain='all'):
 x=SUM[SUM.method.eq(method)&SUM.domain.eq(domain)]
 assert len(x)==1,(method,domain,len(x));return x.iloc[0]

def number(v,p=None):
 if abs(v)>=1000:
  power=int(np.floor(np.log10(abs(v))));coefficient=abs(v)/10.**power
  sign='-' if v<0 else ''
  return rf'${sign}10^{{{power}}}$' if np.isclose(coefficient,1) else rf'${sign}{coefficient:g}\times10^{{{power}}}$'
 return f'{v:g}'

def nse_axis(a,label='NSE',axis='y'):
 if axis=='y':
  a.set_yscale('symlog',linthresh=1,linscale=.8);a.axhline(0,c=GRAY,lw=.7,ls=':');a.set_ylabel(label)
  a.yaxis.set_minor_locator(NullLocator());a.yaxis.set_major_formatter(FuncFormatter(number))
 else:
  a.set_xscale('symlog',linthresh=1,linscale=.8);a.axvline(0,c=GRAY,lw=.7,ls=':');a.set_xlabel(label)
  a.xaxis.set_minor_locator(NullLocator());a.xaxis.set_major_formatter(FuncFormatter(number))

def dots(a,methods,labels,metric='NSE',domains=('all','within'),positions=None,show_points=True):
 x=np.arange(len(methods)) if positions is None else np.asarray(positions)
 data=[]
 for dom,c,m,off,lab in [('all',BLUE,'o',-.105,'All validation WSE'),('within',ORANGE,'^',.105,'Within calibration WSE range')]:
  if dom not in domains:continue
  z=pd.DataFrame([row(k,dom) for k in methods]);med=z[f'{metric}_median'].to_numpy();lo=z[f'{metric}_q25'].to_numpy();hi=z[f'{metric}_q75'].to_numpy()
  if show_points:
   for xx,method in zip(x,methods):
    values=REP[REP.method.eq(method)&REP.domain.eq(dom)].sort_values('fold_id')[metric].to_numpy()
    a.scatter(xx+off+np.linspace(-.035,.035,len(values)),values,s=9,color=c,alpha=.4,lw=0,zorder=2)
  a.errorbar(x+off,med,yerr=np.stack([med-lo,hi-med]),fmt=m,c=c,lw=1.1,capsize=2.1,ms=4.1,label=lab,zorder=4)
  for j,method in enumerate(methods):data.append({'method':method,'domain':dom,'metric':metric,'median':med[j],'q25':lo[j],'q75':hi[j]})
 a.set_xticks(x,labels);a.tick_params(axis='x',labelsize=7.1)
 if metric=='NSE':nse_axis(a,'NSE (across-gauge validation pairs)')
 else:a.set_yscale('log');a.set_ylabel(r'RMSE (m$^3$ s$^{-1}$)');a.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:,.0f}'))
 return data


BOUNDARY = Path(__file__).resolve().parent / 'assets/boundary_geometry.json'
PROJECTION = '+proj=lcc +lat_1=25 +lat_2=47 +lat_0=35 +lon_0=105 +datum=WGS84 +units=m'
MAIN_LATITUDE_MIN = 17.0
INSET_BOUNDS = (105.0, 2.4, 125.5, 25.8)
_MAP_CACHE = None
MAP_QA = []

class PointTable:
    """The original scatter layers use geometry.x/y plus unchanged tabular values."""
    def __init__(self, data):
        self.data = data
    def __getattr__(self, key):
        if key == 'geometry':
            return self
        if key == 'x':
            return self.data['_x']
        if key == 'y':
            return self.data['_y']
        return getattr(self.data, key)

def polygon_parts(g):
    if g.is_empty:
        return
    if isinstance(g, Polygon):
        yield g
    elif isinstance(g, (MultiPolygon, GeometryCollection)):
        for part in g.geoms:
            yield from polygon_parts(part)

def line_parts(g):
    if g.is_empty:
        return
    if g.geom_type in ('LineString', 'LinearRing'):
        yield np.asarray(g.coords)
    elif hasattr(g, 'geoms'):
        for part in g.geoms:
            yield from line_parts(part)

def draw_polygons(ax, geometry, facecolor, edgecolor, linewidth, zorder):
    for polygon in polygon_parts(geometry):
        vertices, codes = [], []
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = list(ring.coords)
            vertices.extend(coords)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(coords)-2) + [MplPath.CLOSEPOLY])
        ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=facecolor,
                              edgecolor=edgecolor, lw=linewidth, zorder=zorder))

def padded_bounds(g, fraction=.026):
    x0, y0, x1, y1 = g.bounds
    px, py = (x1-x0)*fraction, (y1-y0)*fraction
    return [x0-px, x1+px, y0-py, y1+py]

def map_data():
    global _MAP_CACHE
    if _MAP_CACHE is not None:
        return _MAP_CACHE
    records = json.loads(BOUNDARY.read_text(encoding='utf8'))
    land_parts, dash_parts = [], []
    all_lonlat = []
    for record in records:
        # PyShp's polygon conversion handles shapefile exterior/hole ring orientation.
        shp = shapefile.Shape(shapeType=5)
        shp.parts, shp.points = [], []
        for ring in record['rings']:
            shp.parts.append(len(shp.points))
            shp.points.extend(ring)
            all_lonlat.extend(ring)
        geom = make_valid(shape(shp.__geo_interface__))
        (dash_parts if record['attributes'].get('adchar') == 'JD' else land_parts).append(geom)
    land = make_valid(unary_union(land_parts))
    dashes = make_valid(unary_union(dash_parts))
    all_geom = unary_union([land, dashes])
    land_projected = make_valid(transform_geometry(_PROJECT, land))
    dashes_projected = make_valid(transform_geometry(_PROJECT, dashes))
    main_land_ll = unary_union([p for p in polygon_parts(land) if p.bounds[1] >= MAIN_LATITUDE_MIN])
    main_dashes_ll = unary_union([p for p in polygon_parts(dashes) if p.bounds[1] >= MAIN_LATITUDE_MIN])
    main_land = make_valid(transform_geometry(_PROJECT, main_land_ll))
    main_dashes = make_valid(transform_geometry(_PROJECT, main_dashes_ll))
    main_extent = padded_bounds(unary_union([main_land, main_dashes]))
    # Reserve an eastern blank strip so the inset cannot cover coastal gauges.
    main_extent[1] += .23 * (main_extent[1]-main_extent[0])
    # A densified regional rectangle determines a truthful, separately scaled inset.
    ix0, iy0, ix1, iy1 = INSET_BOUNDS
    edge = []
    for xx, yy in [(np.linspace(ix0,ix1,121), np.full(121,iy0)),
                   (np.full(121,ix1), np.linspace(iy0,iy1,121)),
                   (np.linspace(ix1,ix0,121), np.full(121,iy1)),
                   (np.full(121,ix0), np.linspace(iy1,iy0,121))]:
        edge.extend(zip(xx, yy))
    inset_extent = padded_bounds(transform_geometry(_PROJECT, Polygon(edge)), .0)
    # Every supplied boundary vertex must be visible in the main map or the inset.
    vertices = np.asarray(all_lonlat)
    px, py = _PROJECT(vertices[:,0], vertices[:,1])
    def covered(ext):
        return (px >= ext[0]) & (px <= ext[1]) & (py >= ext[2]) & (py <= ext[3])
    coverage = covered(main_extent) | covered(inset_extent)
    assert coverage.all(), f'{(~coverage).sum()} supplied vertices fall outside both frames'
    coords = pd.read_csv(COORDINATES, dtype={'station_id': str})
    coords['_x'], coords['_y'] = _PROJECT(coords.longitude.to_numpy(), coords.latitude.to_numpy())
    assert ((coords._x >= main_extent[0]) & (coords._x <= main_extent[1]) &
            (coords._y >= main_extent[2]) & (coords._y <= main_extent[3])).all()
    assert (coords.latitude >= MAIN_LATITUDE_MIN).all()
    basin_file = BASIN
    basin_crs = CRS.from_wkt(basin_file.with_suffix('.prj').read_text())
    basin_project = Transformer.from_crs(basin_crs, PROJECTION, always_xy=True).transform
    basin_lines = []
    reader = shapefile.Reader(str(basin_file))
    for shp in reader.iterShapes():
        geom = make_valid(transform_geometry(basin_project, make_valid(shape(shp.__geo_interface__))))
        # Retain the original basin context, clipped to the supplied national land polygon.
        geom = make_valid(geom.simplify(200, preserve_topology=True)).intersection(land_projected, grid_size=.1)
        basin_lines.extend(line_parts(geom.boundary))
    _MAP_CACHE = {
        'coords': coords, 'land': land_projected, 'dashes': dashes_projected,
        'main_land': main_land, 'main_dashes': main_dashes,
        'main': main_extent, 'inset': inset_extent, 'basins': basin_lines,
        'boundary_vertex_count': len(vertices), 'covered_vertex_count': int(coverage.sum()),
        'station_count': len(coords), 'station_min_latitude': float(coords.latitude.min())}
    # Validate the geometry actually drawn in both frames, not just its vertices.
    main_box=box(main_extent[0],main_extent[2],main_extent[1],main_extent[3])
    inset_box=box(inset_extent[0],inset_extent[2],inset_extent[1],inset_extent[3])
    visible_land=unary_union([main_land.intersection(main_box),land_projected.intersection(inset_box)])
    visible_dashes=unary_union([main_dashes.intersection(main_box),dashes_projected.intersection(inset_box)])
    missing_land=land_projected.difference(visible_land)
    missing_dashes=dashes_projected.difference(visible_dashes)
    assert missing_land.area < .01 and missing_dashes.area < .01
    assert missing_dashes.length < .01
    _MAP_CACHE['geometry_coverage']={'missing_land_area_m2':missing_land.area,
        'missing_boundary_area_m2':missing_dashes.area,'missing_boundary_length_m':missing_dashes.length}

    return _MAP_CACHE

def geography():
    m = map_data()
    return PointTable(m['coords']), None, m['main']

def basemap(a):
    m = map_data()
    draw_polygons(a, m['main_land'], '#fafbfb', 'none', 0, 0)
    a.add_collection(LineCollection(m['basins'], colors='#d4dce0', linewidths=.25, zorder=.5))
    # The supplied national boundary, including all islands, is drawn above basin context.
    draw_polygons(a, m['main_land'], 'none', '#687984', .48, 1)
    draw_polygons(a, m['main_dashes'], '#687984', '#687984', .32, 1)
    g = m['coords']
    a.scatter(g._x, g._y, s=5, c='#cdd3d6', lw=0, zorder=2)
    a.set(xlim=m['main'][:2], ylim=m['main'][2:])
    a.set_aspect('equal')
    a.axis('off')
    inset = a.inset_axes([.835, .025, .16, .42], zorder=10)
    inset.set_facecolor('white')
    draw_polygons(inset, m['land'], '#fafbfb', '#687984', .4, 0)
    draw_polygons(inset, m['dashes'], '#687984', '#687984', .34, 1)
    inset.set(xlim=m['inset'][:2], ylim=m['inset'][2:])
    inset.set_aspect('equal', adjustable='box')
    inset.set_anchor('SE')
    inset.set_xticks([]); inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True); spine.set_color('#83909B'); spine.set_linewidth(.55)
    inset.text(.5, 1.025, 'South China Sea', transform=inset.transAxes,
               ha='center', va='bottom', fontsize=4.6, color=INK)
    a._national_map = True
    a._south_inset = inset
    MAP_QA.append({'figure': a.figure.number, 'main_extent': m['main'], 'inset_extent': m['inset'],
                   'boundary_vertex_count': m['boundary_vertex_count'],
                   'covered_vertex_count': m['covered_vertex_count'],
                   'station_count': m['station_count']})

def xy(d):
    return PointTable(map_data()['coords'].merge(d, on='station_id', validate='one_to_one'))


def concept():
 """Retain the latest embedded a/b exactly; replace only panel c."""
 from PIL import Image
 source=SOURCE_FIGURE
 image_path=FIG/'source_Figure_1.png'
 shutil.copy2(source,image_path)
 original=Image.open(image_path);top=original.crop((0,0,1650,1420))
 f=plt.figure(figsize=(7.2,8.64));a=f.add_axes([0,0,1,1]);a.set(xlim=(0,1650),ylim=(1980,0));a.axis('off')
 a.imshow(top,extent=(0,1650,1420,0),interpolation='none')
 def tx(x,y,t,size=8.2,color=INK,bold=False,ha='left'):
  return a.text(x,y,t,fontsize=size,c=color,weight='bold' if bold else 'normal',ha=ha,va='top',linespacing=1.3)
 tx(90,1434,'c  Calibration and validation',10.5,bold=True)
 tx(240,1505,'5 WSE strata · 5 validation folds',8.6,bold=True)
 tx(240,1547,'About 80% calibration / 20% validation',7.6,GRAY)
 for j in range(5):tx(266+j*109,1594,str(j+1),7.4,GRAY,ha='center')
 for fold in range(5):
  y=1641+36*fold;tx(105,y-11,f'Fold {fold+1}',7.2,GRAY)
  for j in range(5):
   xs=230+j*109+np.arange(5)*18
   for k,x in enumerate(xs):a.plot(x,y,'o',ms=3.1,mec=ORANGE if k==fold else BLUE,mfc='white' if k==fold else BLUE,mew=.75)
 a.annotate('',xy=(765,1835),xytext=(213,1835),arrowprops={'arrowstyle':'->','color':GRAY,'lw':.8})
 tx(485,1849,'Increasing WSE',7.7,GRAY,ha='center')
 for x,lab,col,fill in [(270,'Calibration',BLUE,BLUE),(490,'Validation',ORANGE,'white')]:
  a.plot(x,1900,'o',ms=3.4,mec=col,mfc=fill,mew=.8);tx(x+25,1888,lab,7.8)
 tx(1195,1505,'Calibration WSE range',8.6,bold=True,ha='center')
 a.plot([965,1520],[1715,1715],c=GRAY,lw=.8);a.plot([1110,1350],[1715,1715],c=BLUE,lw=3.7)
 for x in [1110,1350]:a.plot([x,x],[1640,1742],c=GRAY,lw=.65,ls='--')
 for x,t,col in [(1020,'Below',ORANGE),(1230,'Within',BLUE),(1450,'Above',ORANGE)]:tx(x,1760,t,8.1,col,ha='center')
 tx(1230,1835,'Fixed W(h) and hydraulic parameters',7.5,GRAY,ha='center')
 tx(1230,1888,'Each observation is validated once',7.8,TEAL,ha='center')
 tx(90,1950,'Across-gauge NSE and RMSE: median ± IQR/2 over five folds. Gauge-level scores use all out-of-fold predictions.',7.2,GRAY)
 save(f,1)
 REPORT['Figure_1']={'original_ab_source':str(source),'original_ab_image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'ab_preserved_crop':[0,0,1650,1420],'validation':'standard fivefold; each observation validated once'}

def support():
 q=pd.read_csv(QC_SUMMARY,dtype={'station_id':str})
 ids=geography()[0].station_id;q=q[q.station_id.isin(ids)]
 # The latest manuscript intentionally uses only panels a/b.  Use a true
 # two-panel canvas instead of retaining the old Word crop of a third panel.
 f=plt.figure(figsize=(7.2,3.7));gs=f.add_gridspec(1,2,left=.085,right=.98,top=.88,bottom=.18,wspace=.63)
 a=f.add_subplot(gs[0,0]);map_title(f,gs[0,0],'a','Gauges available for calibration');basemap(a);g=xy(q)
 sc=a.scatter(g.geometry.x,g.geometry.y,c=g.retained_observations,s=15,norm=LogNorm(vmin=5,vmax=q.retained_observations.max()),cmap='viridis',edgecolor='white',lw=.2,zorder=3)
 cb=f.colorbar(sc,ax=a,orientation='horizontal',fraction=.045,pad=.028);cb.set_label('QC-retained overpasses per gauge',fontsize=7.6)
 a.text(.02,.02,'542 calibration gauges',transform=a.transAxes,fontsize=7.7)
 b=f.add_subplot(gs[0,1]);panel(b,'b','Data retained at each stage')
 labs=['Satellite archive','Complete SWOT–Q pairs','After satellite QC','At calibration gauges','Fivefold CV support'];vals=[78643,10756,8859,8268,7615]
 b.barh(np.arange(5)[::-1],vals,color=[LIGHT,GRAY,BLUE,TEAL,PURPLE],height=.58);b.set_xlim(0,101000);b.set_yticks(np.arange(5)[::-1],labs);b.tick_params(axis='y',labelsize=7.2);b.set_xticks([0,40000,80000],['0','40,000','80,000']);b.set_xlabel('Unique gauge overpasses')
 for y,v in zip(np.arange(5)[::-1],vals):b.text(v+1600,y,f'{v:,}',fontsize=7.4,va='center')
 save(f,2,[{'stage':label,'unique_gauge_overpasses':count} for label,count in zip(labs,vals)])
 REPORT['Figure_2']={'panels':['a','b'],'figure_inches':[7.2,3.7],'Word_crop_required':False,'CV_gauges':422,'CV_observations':7615}

def width():
 zpath=WSE_Z_PATH
 z=pd.read_csv(zpath,dtype={'station_id':str});rcol=next(c for c in ['pearson_r','r','WSE_Z_pearson_r'] if c in z)
 vals=z[rcol].dropna().sort_values();countcol=next((c for c in ['n_pairs','nobs','n'] if c in z),None)
 f,ax=plt.subplots(2,2,figsize=(7.2,6.0));f.subplots_adjust(left=.105,right=.985,bottom=.18,top=.925,hspace=.72,wspace=.43)
 a,b,c,d=ax.ravel();panel(a,'a','SWOT WSE and gauge water level')
 a.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',c=BLUE,lw=1.5);a.set(xlim=(-1,1),ylim=(0,1.02),xlabel='Gauge-level Pearson r',ylabel='Cumulative gauge fraction');a.set_xticks([-1,-.5,0,.5,1]);a.axvline(0,c=GRAY,ls=':',lw=.8)
 pairs=int(z.loc[z[rcol].notna(),countcol].sum()) if countcol else None
 a.text(.03,.94,f'{len(vals)} gauges'+(f'; {pairs:,} pairs' if pairs is not None else '')+f'\nr = {vals.median():.3f} ± {(vals.quantile(.75)-vals.quantile(.25))/2:.3f}',transform=a.transAxes,va='top',fontsize=7.6)
 panel(b,'b','Calibration WSE–width relationship')
 p=pd.read_csv(CV/'tables/parameter_estimates.csv',usecols=['station_id','method','fold_id','width_curve_parameters_json'],dtype={'station_id':str})
 example=p[p.station_id.eq('30700300')&p.method.eq('full')&p.fold_id.eq(2)].iloc[0]
 q=json.loads(example.width_curve_parameters_json);xx=np.array(q['curve_wse'])-min(q['curve_wse'])
 b.scatter(xx,q['raw_width'],s=15,c=GRAY,label='Observed calibration width');b.plot(xx,q['curve_width'],c=BLUE,lw=1.5,label='Corrected relationship');b.set(xlabel='WSE above lowest anchor (m)',ylabel='Width (m)',ylim=(0,max(q['raw_width'])*1.35));b.legend(frameon=False,fontsize=6.8,loc='upper left');b.text(.98,.95,'Gauge 30700300 · fold 3',transform=b.transAxes,ha='right',va='top',fontsize=6.6,color=GRAY)
 methods=['observed_fixed','corrected_fixed','no_monotonic','full'];xs=np.array([0.,1.,2.6,3.6]);data=[]
 for target,metric,k,title in [(c,'NSE','c','Effect of width correction on NSE'),(d,'RMSE','d','Effect of width correction on RMSE')]:
  panel(target,k,title);data+=dots(target,methods,['Observed\nwidth','Corrected\nwidth','Observed\nwidth','Corrected\nwidth'],metric,positions=xs)
  target.set_xlim(-.48,4.08);target.axvline(1.8,c=LIGHT,ls=':',lw=.8)
  for x,label in [(.5,'Fixed hydraulic parameters'),(3.1,'Jointly calibrated parameters')]:target.text(x,-.30,label,transform=target.get_xaxis_transform(),ha='center',va='top',fontsize=6.8)
  if metric=='NSE':target.set_yticks([-1e6,-1e4,-100,-1,0,1])
  else:target.set_yticks([1000,10000,100000,1000000])
 handles=[Line2D([],[],marker='o',ls='none',c=BLUE,ms=4.5,label='All validation WSE'),Line2D([],[],marker='^',ls='none',c=ORANGE,ms=4.5,label='Within calibration WSE range')]
 f.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.01),frameon=False,ncol=2,fontsize=7.2)
 save(f,3,data);REPORT['Figure_3']={'WSE_Z_source':str(zpath),'example_gauge':'30700300','example_fold_display':3}

def calibration():
 f,ax=plt.subplots(2,2,figsize=(7.2,6.0));f.subplots_adjust(left=.105,right=.985,bottom=.12,top=.92,hspace=.78,wspace=.42)
 a,b,c,d=ax.ravel();data=[]
 for target,metric,title,k in [(a,'NSE','Across-gauge validation NSE','a'),(b,'RMSE','Across-gauge validation RMSE','b')]:
  panel(target,k,title)
  for dom,col,m,lab in [('all',BLUE,'o','All validation WSE'),('within',ORANGE,'^','Within calibration WSE range')]:
   z=REP[REP.method.eq('full')&REP.domain.eq(dom)].sort_values('fold_id');target.plot(z.fold_id+1,z[metric],marker=m,c=col,lw=.8,ms=4,label=lab)
   data.extend(z[['fold_id','method','domain',metric]].rename(columns={metric:'value'}).assign(metric=metric).to_dict('records'))
  target.set_xticks(range(1,6));target.set_xlabel('WSE-stratified CV fold')
  if metric=='NSE':nse_axis(target);target.set_yticks([-10,-1,0,1]);target.text(.03,.03,'All out-of-fold NSE = −0.706',transform=target.transAxes,fontsize=6.7,color=GRAY)
  else:target.set_yscale('log');target.set_ylabel(r'RMSE (m$^3$ s$^{-1}$)');target.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:,.0f}'))
 a.legend(frameon=False,fontsize=6.5,loc='lower left',bbox_to_anchor=(0,.14))
 methods=['n_only_fixed_geometry','fixed_reference_n','beta_fixed_1','full'];labels=['Calibrated n\nFixed geometry','Fixed\nreference n','Fixed β = 1','Full model']
 panel(c,'c','Hydraulic parameter calibration');data+=dots(c,methods,labels);c.set_yticks([-1e4,-100,-1,0,1])
 panel(d,'d','Absolute discharge error');data+=dots(d,methods,labels,'RMSE');d.set_yticks([500,1000,5000,10000,50000])
 save(f,4,data)

def errors():
 evid=ERROR_EVIDENCE
 concentration=evid/'screening_concentration_by_fold.csv';screen=evid/'screening_metrics_summary.csv'
 if not concentration.exists() or not screen.exists():raise FileNotFoundError('Waiting for shared full-model-error screening evidence')
 f,ax=plt.subplots(2,2,figsize=(7.2,6.0));f.subplots_adjust(left=.11,right=.985,bottom=.12,top=.92,hspace=.78,wspace=.43)
 a,b,c,d=ax.ravel();panel(a,'a','NSE within and beyond calibration');domains=['within','above','below']
 data=[]
 for i,dom in enumerate(domains):
  r=row('full',dom);values=REP[REP.method.eq('full')&REP.domain.eq(dom)].sort_values('fold_id').NSE.to_numpy()
  a.scatter(i+np.linspace(-.06,.06,5),values,c=BLUE,s=11,alpha=.45,lw=0)
  a.errorbar(i,r.NSE_median,yerr=[[r.NSE_median-r.NSE_q25],[r.NSE_q75-r.NSE_median]],fmt='o',c=BLUE,lw=1.1,capsize=2.5,ms=4)
  a.text(i,.045,f'n≈{r.nobs_median:g}',transform=a.get_xaxis_transform(),ha='center',fontsize=7)
  data.append({'panel':'a','domain':dom,'metric':'NSE','median':r.NSE_median,'q25':r.NSE_q25,'q75':r.NSE_q75,'nobs_median':r.nobs_median})
 nse_axis(a,'NSE (across-gauge validation pairs)');a.set_xticks(range(3),['Within','Above','Below']);a.set_yticks([-100,-10,-1,0,1])
 panel(b,'b','Squared error in the largest residuals');z=pd.read_csv(concentration);z=z[z.rule.eq('absolute_top1')]
 fcol='fold_id' if 'fold_id' in z else 'repeat_id';scol=next(c for c in ['tail_SSE_fraction','removed_SSE_fraction','SSE_fraction'] if c in z)
 data.extend(z.assign(panel='b').to_dict('records'))
 b.bar(z[fcol]+1,z[scol]*100,color=BLUE,width=.6);b.set(ylim=(0,105),xlabel='WSE-stratified CV fold',ylabel='Squared error contributed (%)');b.set_xticks(range(1,6));b.text(.03,.96,'Residuals above the 99th percentile',transform=b.transAxes,va='top',fontsize=7.3)
 panel(c,'c','Error with distance beyond calibration');q=pd.read_csv(CV/'tables/extrapolation_factor_error_OOF_summary.csv')
 for side,col,marker,off,lab in [('above',BLUE,'o',-.05,'Above range'),('below',ORANGE,'^',.05,'Below range')]:
  z=q[q.domain.eq(side)].reset_index(drop=True);xx=np.arange(len(z))+off
  for i,r in z.iterrows():
   c.vlines(xx[i],max(r.q25,1),min(r.q75,45),color=col,lw=.9)
   if r.q75>45:c.annotate('',xy=(xx[i],49),xytext=(xx[i],30),arrowprops={'arrowstyle':'->','color':col,'lw':.9})
   c.text(i,.035 if side=='above' else .115,f'{int(r.N)}',transform=c.get_xaxis_transform(),ha='center',fontsize=6.1,color=col)
  c.plot(xx,z['median'],c=col,marker=marker,ms=4,lw=.9,label=lab)
  data.extend(z.assign(panel='c').to_dict('records'))
 c.set_yscale('log');c.set_ylim(1,50);c.set_yticks([1,2,5,10,20,50],['1','2','5','10','20','50']);c.yaxis.set_minor_formatter(NullFormatter());c.set_xticks(range(4),['(0,0.25]','(0.25,0.5]','(0.5,1]','>1']);c.tick_params(axis='x',labelsize=6.7);c.set_xlabel('Boundary distance / calibration WSE span',fontsize=7.8);c.set_ylabel('Gauge-level factor error');c.legend(frameon=False,fontsize=6.7,loc='upper left')
 panel(d,'d','NSE after residual screening');q=pd.read_csv(screen)
 screening_folds=pd.read_csv(evid/'screening_metrics_by_fold.csv')
 screening_folds=screening_folds[screening_folds.model.eq('Hydraulic')&screening_folds.rule.isin(['baseline','absolute_top1'])&screening_folds.domain.isin(['all','within'])]
 q=q[q.model.eq('Hydraulic')&q.metric.eq('NSE')&q.rule.isin(['baseline','absolute_top1'])&q.domain.isin(['all','within'])]
 for dom,col,marker,off,lab in [('all',BLUE,'o',-.08,'All validation WSE'),('within',ORANGE,'^',.08,'Within calibration WSE range')]:
  z=q[q.domain.eq(dom)].set_index('rule').loc[['baseline','absolute_top1']]
  for xpos,rule in enumerate(['baseline','absolute_top1']):
   fold_points=screening_folds[screening_folds.domain.eq(dom)&screening_folds.rule.eq(rule)].sort_values('fold_id')
   assert len(fold_points)==5,(dom,rule,len(fold_points))
   d.scatter(xpos+off+np.linspace(-.035,.035,5),fold_points.NSE,s=11,marker=marker,color=col,alpha=.45,lw=0,zorder=2)
   data.extend(fold_points.assign(panel='d',plot_statistic='individual_fold_score',metric='NSE').to_dict('records'))
  d.errorbar(np.arange(2)+off,z['median'],yerr=np.stack([z['median']-z.q25,z.q75-z['median']]),fmt=marker,c=col,lw=1,capsize=2.5,label=lab)
  data.extend(z.reset_index().assign(panel='d').to_dict('records'))
 nse_axis(d,'NSE (across-gauge validation pairs)');d.set_ylim(min(-10,float(screening_folds.NSE.min())*1.1),1.12);d.set_yticks([-10,-1,0,1]);d.set_xticks([0,1],['Original','Above 99th percentile\nremoved']);d.tick_params(axis='x',labelsize=7);d.set_xlim(-.3,1.35);d.legend(frameon=False,fontsize=6.5,loc='lower right')
 REPORT['Figure_5d']={'individual_fold_points':len(screening_folds),'minimum_fold_NSE':float(screening_folds.NSE.min()),'y_limits':list(d.get_ylim()),'y_ticks':[-10,-1,0,1]}
 save(f,5,data)

def products():
 pred=pd.read_csv(CV/'private/products_matched_observations_RESTRICTED.csv',usecols=['station_id','fold_id','obs_id','sos_consensus_q','sos_metroman_q','sos_momma_q','sos_sic4dvar_q'],dtype={'station_id':str,'obs_id':str})
 cols=['sos_consensus_q','sos_metroman_q','sos_momma_q','sos_sic4dvar_q'];pred=pred[np.isfinite(pred[cols]).all(axis=1)&pred[cols].gt(0).all(axis=1)]
 assert len(pred)==794 and pred.station_id.nunique()==108,(len(pred),pred.station_id.nunique())
 f=plt.figure(figsize=(7.2,5.4));gs=f.add_gridspec(2,2,left=.10,right=.98,bottom=.12,top=.92,wspace=.48,hspace=.68,height_ratios=[1.1,1])
 a=f.add_subplot(gs[0,:]);map_title(f,gs[0,:],'a','Gauges in the common four-product comparison');basemap(a);g=xy(pred[['station_id']].drop_duplicates());a.scatter(g.geometry.x,g.geometry.y,s=14,c=BLUE,edgecolor='white',lw=.3,zorder=3)
 counts=pred.groupby('fold_id').agg(n=('obs_id','size'),g=('station_id','nunique'))
 f.text(.10,.715,f'108 gauges\n794 unique validation overpasses\nEach fold: {counts.n.min()}–{counts.n.max()} pairs\nat {counts.g.min()}–{counts.g.max()} gauges',fontsize=7.7,va='top',linespacing=1.5)
 q=pd.read_csv(CV/'tables/products_fold_summary.csv');q=q[q.support.eq('common_four')&q.domain.eq('all')]
 rep=pd.read_csv(CV/'tables/products_metrics_by_fold.csv');rep=rep[rep.support.eq('common_four')&rep.domain.eq('all')]
 models=['Local','Consensus','MetroMan','MOMMA','SIC4DVar'];colors=[BLUE,GRAY,ORANGE,TEAL,PURPLE];data=[]
 for spec,metric,k,title in [(gs[1,0],'NSE','b','NSE on identical validation pairs'),(gs[1,1],'RMSE','c','RMSE on identical validation pairs')]:
  ax=f.add_subplot(spec);panel(ax,k,title);z=q[q.metric.eq(metric)].set_index('model').loc[models]
  for i,(name,col) in enumerate(zip(models,colors)):
   r=z.loc[name];values=rep[rep.model.eq(name)][metric].to_numpy();ax.scatter(i+np.linspace(-.05,.05,len(values)),values,s=11,color=col,alpha=.45,lw=0)
   ax.errorbar(i,r['median'],yerr=[[r['median']-r.q25],[r.q75-r['median']]],fmt='o',c=col,ms=4.3,capsize=2.3,lw=1.1)
   data.append({'model':name,'metric':metric,'median':r['median'],'q25':r.q25,'q75':r.q75})
  ax.set_xticks(range(5),models,rotation=25,ha='right');ax.tick_params(axis='x',labelsize=7)
  if metric=='NSE':nse_axis(ax,'NSE (across-gauge validation pairs)')
  else:ax.set_ylim(bottom=0);ax.set_ylabel(r'RMSE (m$^3$ s$^{-1}$)')
 save(f,6,data)
 REPORT['Figure_6']={'common_gauges':108,'common_observations':794,'fold_counts':counts.to_dict('index')}

def gauge_analysis():
 f=plt.figure(figsize=(7.2,6.4));gs=f.add_gridspec(2,2,left=.09,right=.985,bottom=.085,top=.92,wspace=.45,hspace=.59)
 local=pd.read_csv(CV/'tables/station_analysis_gauge_metrics.csv',dtype={'station_id':str});local=local[local.support.eq('native')&local.domain.eq('all')&local.method.eq('full')&local.NSE.notna()]
 assert len(local)==418,len(local)
 a=f.add_subplot(gs[0,0]);map_title(f,gs[0,0],'a','Out-of-fold NSE at individual gauges');basemap(a);g=xy(local)
 cmap=ListedColormap(['#965E43','#D8B48B','#EADCC7','#88C3CD',BLUE]);norm=BoundaryNorm([-1e30,-10,-1,0,.5,1.01],5)
 size=lambda s: 9+3*np.sqrt(np.asarray(s,float))
 a.scatter(g.geometry.x,g.geometry.y,c=g.NSE,s=size(g.nobs),cmap=cmap,norm=norm,edgecolor='white',lw=.2,zorder=3)
 cb=f.colorbar(ScalarMappable(norm=BoundaryNorm(range(6),5),cmap=cmap),ax=a,orientation='horizontal',fraction=.05,pad=.026);cb.set_ticks(np.arange(5)+.5,labels=['<−10','−10–−1','−1–0','0–0.5','≥0.5']);cb.ax.tick_params(labelsize=6.3);cb.set_label('Gauge-level out-of-fold NSE',fontsize=7.5)
 a.text(.02,1.065,'418 eligible gauges\nGray: criterion unmet / outside CV\nSymbol size: validation observations',transform=a.transAxes,fontsize=6.6,va='top')
 b=f.add_subplot(gs[0,1]);panel(b,'b','Distribution across eligible gauges');v=np.sort(local.NSE.to_numpy());b.step(v,np.arange(1,len(v)+1)/len(v),where='post',c=BLUE,lw=1.5)
 b.set_xscale('symlog',linthresh=.3,linscale=.7);b.set_xlim(v.min()*1.10,1.02);b.set_ylim(0,1.03);b.set_xticks([-1e6,-1e3,-1,0,1]);b.xaxis.set_major_formatter(FuncFormatter(number));b.xaxis.set_minor_locator(NullLocator());b.axvline(0,c=GRAY,lw=.8,ls=':');b.set_xlabel('Gauge-level out-of-fold NSE\n(symmetric-log scale)');b.set_ylabel('Cumulative gauge fraction')
 med=local.NSE.median();half=(local.NSE.quantile(.75)-local.NSE.quantile(.25))/2;positive=local.NSE.gt(0).mean()*100
 b.text(.03,.96,f'418 gauges\nNSE = {med:.3f} ± {half:.3f}\nPositive NSE at {positive:.1f}% of gauges',transform=b.transAxes,va='top',fontsize=7.4)
 paired=pd.read_csv(CV/'tables/products_OOF_gauge_metrics.csv',dtype={'station_id':str});paired=paired[paired.support.eq('product_specific')&paired.domain.eq('all')]
 names=['Consensus','MetroMan','MOMMA','SIC4DVar'];diff=[];stats=[]
 for name in names:
  z=paired[paired['product'].eq(name)].pivot(index='station_id',columns='model',values=['NSE','nobs']);z=z.dropna(subset=[('NSE','Local'),('NSE',name)])
  delta=z[('NSE','Local')]-z[('NSE',name)];tmp=pd.DataFrame({'station_id':delta.index,'product':name,'delta':delta.values,'nobs':z[('nobs','Local')].to_numpy()});diff.append(tmp)
  stats.append({'product':name,'N':len(delta),'median':delta.median(),'q25':delta.quantile(.25),'q75':delta.quantile(.75),'positive_fraction':delta.gt(0).mean()})
 diff=pd.concat(diff,ignore_index=True);stats=pd.DataFrame(stats).set_index('product')
 c=f.add_subplot(gs[1,0]);panel(c,'c','Paired gauge-level improvement');pos=c.get_position();c.set_position([pos.x0+.095,pos.y0,pos.width-.095,pos.height])
 for i,name in enumerate(names):
  r=stats.loc[name];c.errorbar(r['median'],i,xerr=[[r['median']-r.q25],[r.q75-r['median']]],fmt='o',c=BLUE,lw=1.1,capsize=2.5,ms=4)
 c.set_yticks(range(4),[f'{n} (n={int(stats.loc[n,"N"])})' for n in names]);c.tick_params(axis='y',labelsize=7);c.invert_yaxis();nse_axis(c,'Local NSE − product NSE\n(symmetric-log scale)','x');c.set_ylabel('');c.grid(axis='x',c=LIGHT,lw=.5)
 d=f.add_subplot(gs[1,1]);map_title(f,gs[1,1],'d','Local comparison with SIC4DVar');basemap(d);z=diff[diff['product'].eq('SIC4DVar')];g=xy(z)
 cm=ListedColormap(['#9c5a40','#e2c7af','#d7e9eb','#75b6c5',BLUE]);nm=BoundaryNorm([-1e30,-1,0,1,10,1e30],5)
 d.scatter(g.geometry.x,g.geometry.y,c=g.delta,s=size(g.nobs),cmap=cm,norm=nm,edgecolor='white',lw=.2,zorder=3)
 cb=f.colorbar(ScalarMappable(norm=BoundaryNorm(range(6),5),cmap=cm),ax=d,orientation='horizontal',fraction=.05,pad=.026);cb.set_ticks(np.arange(5)+.5,labels=['<−1','−1–0','0–1','1–10','≥10']);cb.ax.tick_params(labelsize=6.3);cb.set_label('Paired NSE difference at each gauge',fontsize=7.5)
 d.text(.02,.05,f'{len(z)} paired gauges\nPositive = local improvement',transform=d.transAxes,fontsize=6.8)
 save(f,7,stats.reset_index().to_dict('records'));diff.to_csv(FIG/'Figure_7_paired_gauge_derived_scores.csv',index=False,encoding='utf-8-sig')
 REPORT['Figure_7']={'NSE_eligible_gauges':len(local),'median_NSE':med,'half_IQR_NSE':half,'positive_NSE_percent':positive,'product_specific_paired':stats.reset_index().to_dict('records')}


def _dependencies(need_maps):
    """Use the active Python environment, without private runtime fallbacks."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullLocator, NullFormatter
    exposed = {name: value for name, value in locals().items() if name in {
        'np','pd','plt','BoundaryNorm','ListedColormap','LogNorm',
        'ScalarMappable','Line2D','FuncFormatter','NullLocator','NullFormatter'}}
    plt.rcParams.update({'font.family':'Arial','font.size':8.4,'axes.labelsize':8.5,'axes.titlesize':9,
     'xtick.labelsize':7.4,'ytick.labelsize':7.5,'legend.fontsize':7.2,'axes.spines.top':False,
     'axes.spines.right':False,'axes.edgecolor':GRAY,'axes.linewidth':.7,'text.color':INK,
     'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'svg.fonttype':'none',
     'svg.hashsalt':'swot-fivefold-20261019','savefig.facecolor':'white','figure.facecolor':'white'})
    if need_maps:
        import shapefile
        from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection, box
        from shapely.ops import unary_union, transform as transform_geometry
        from shapely.validation import make_valid
        from pyproj import CRS, Transformer
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath
        from matplotlib.collections import LineCollection
        exposed.update({name:value for name,value in locals().items() if name in {
            'shapefile','shape','Polygon','MultiPolygon','GeometryCollection','box',
            'unary_union','transform_geometry','make_valid','CRS','Transformer',
            'PathPatch','MplPath','LineCollection'}})
        exposed['_PROJECT'] = Transformer.from_crs(4326, PROJECTION, always_xy=True).transform
    return exposed


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, help='Local fivefold run directory containing tables/ and private/')
    p.add_argument('--evidence', type=Path, help='Local WSE/width evidence directory containing private/WSE_Z_gauge_metrics_RESTRICTED.csv')
    p.add_argument('--error-evidence', type=Path, help='Local screening_* CSV directory for Figure 5; defaults to errors/ beside --evidence')
    p.add_argument('--coordinates', type=Path, help='External study_coordinates.csv with station_id, latitude, longitude')
    p.add_argument('--qc-summary', type=Path, help='External global_qc_station_summary.csv for Figure 2')
    p.add_argument('--basin', type=Path, help='External basin .shp; .shx, .dbf and .prj companions are required')
    p.add_argument('--source-figure', type=Path, help='External source_Figure_1.png whose original a/b panels are retained')
    p.add_argument('--boundary-geometry', type=Path, default=BOUNDARY, help='National boundary geometry; defaults to the supplied asset')
    p.add_argument('--output', type=Path, required=True, help='New empty LOCAL figure-output directory')
    p.add_argument('--figures', type=int, choices=range(1,8), nargs='+', default=list(range(1,8)), help='Manuscript figure numbers to rebuild (default: 1 2 3 4 5 6 7)')
    return p


def _validate_inputs(a):
    """Return validated input paths using the standard library only."""
    selected=set(a.figures)
    required=[]
    def file(value,label):
        if value is None:raise ValueError(label+' is required for the selected figures')
        path=Path(value).resolve()
        if not path.is_file():raise ValueError('Missing '+label+': '+str(path))
        required.append(path)
        return path
    def directory(value,label):
        if value is None:raise ValueError(label+' is required for the selected figures')
        path=Path(value).resolve()
        if not path.is_dir():raise ValueError('Missing '+label+': '+str(path))
        return path
    out=a.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('--output must be a new or empty local directory: '+str(out))
    paths={'FIG':out,'CV':a.root.resolve() if a.root else None,'EVIDENCE':a.evidence.resolve() if a.evidence else None,
           'ERROR_EVIDENCE':None,'WSE_Z_PATH':None,'COORDINATES':None,'QC_SUMMARY':None,
           'BASIN':None,'SOURCE_FIGURE':None,'BOUNDARY':a.boundary_geometry.resolve()}
    if 1 in selected:paths['SOURCE_FIGURE']=file(a.source_figure,'--source-figure')
    table_files=set()
    if selected & {3,4,5}:table_files.update(['cross_gauge_summary.csv','cross_gauge_metrics_by_split.csv'])
    if 3 in selected:
        table_files.add('parameter_estimates.csv')
        ev=directory(a.evidence,'--evidence');paths['EVIDENCE']=ev
        z=ev/'private/WSE_Z_gauge_metrics_RESTRICTED.csv'
        if not z.is_file():
            candidates=sorted(p for p in ev.rglob('*.csv') if all(k in p.name.lower() for k in ['wse','z','gauge']))
            if not candidates:raise ValueError('Missing WSE–Z gauge evidence under '+str(ev))
            z=candidates[0]
        paths['WSE_Z_PATH']=file(z,'WSE–Z gauge evidence')
    if 5 in selected:
        table_files.add('extrapolation_factor_error_OOF_summary.csv')
        errors=a.error_evidence or (a.evidence.resolve().parent/'errors' if a.evidence else None)
        errors=directory(errors,'--error-evidence (or errors/ beside --evidence)')
        paths['ERROR_EVIDENCE']=errors
        for name in ['screening_concentration_by_fold.csv','screening_metrics_summary.csv','screening_metrics_by_fold.csv']:
            file(errors/name,'screening evidence '+name)
    if 6 in selected:table_files.update(['products_fold_summary.csv','products_metrics_by_fold.csv'])
    if 7 in selected:table_files.update(['station_analysis_gauge_metrics.csv','products_OOF_gauge_metrics.csv'])
    if table_files or 6 in selected:
        cv=directory(a.root,'--root');paths['CV']=cv
        for name in sorted(table_files):file(cv/'tables'/name,'fivefold table '+name)
        if 6 in selected:file(cv/'private/products_matched_observations_RESTRICTED.csv','local matched-product observations')
    if selected & {2,6,7}:
        paths['COORDINATES']=file(a.coordinates,'--coordinates')
        paths['BASIN']=file(a.basin,'--basin')
        if paths['BASIN'].suffix.lower()!='.shp':raise ValueError('--basin must name a .shp file')
        for suffix in ['.shx','.dbf','.prj']:file(paths['BASIN'].with_suffix(suffix),'basin companion '+suffix)
        file(paths['BOUNDARY'],'national boundary geometry')
    if 2 in selected:paths['QC_SUMMARY']=file(a.qc_summary,'--qc-summary')
    return paths,required


def main(argv=None):
    p=parser();a=p.parse_args(argv)
    a.figures=list(dict.fromkeys(a.figures))
    try:paths,inputs=_validate_inputs(a)
    except (ValueError,OSError) as exc:p.error(str(exc))
    try:imports=_dependencies(bool(set(a.figures)&{2,6,7}))
    except ImportError as exc:p.error('Missing plotting dependency in the active Python environment: '+str(exc))
    globals().update(imports);globals().update(paths)
    globals()['REPORT']={}
    globals()['_MAP_CACHE']=None
    globals()['MAP_QA']=[]
    if set(a.figures)&{3,4,5}:
        summary=pd.read_csv(CV/'tables/cross_gauge_summary.csv')
        reps=pd.read_csv(CV/'tables/cross_gauge_metrics_by_split.csv')
        globals()['SUM']=summary[summary.scheme.eq('wse_stratified_5fold')&summary.support.eq('all_methods_common')]
        globals()['REP']=reps[reps.scheme.eq('wse_stratified_5fold')&reps.support.eq('all_methods_common')]
    FIG.mkdir(parents=True,exist_ok=True)
    actions={1:concept,2:support,3:width,4:calibration,5:errors,6:products,7:gauge_analysis}
    for number in a.figures:actions[number]()
    (FIG/'map_coverage_QA.json').write_text(json.dumps(MAP_QA,indent=2),encoding='utf8')
    (FIG/'figure_build_report.json').write_text(json.dumps(REPORT,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({'status':'PASS','figures':a.figures,'output':str(FIG),'numerical_model_refitted':False}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
