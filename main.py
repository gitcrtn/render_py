# render_py
# Keita Yamada
# 2014.09.13

import numpy
import Image
import math
import random
	
class ImageBuffer(object):
	def __init__(self,x,y):
		self.x = y
		self.y = x
		self.instance = numpy.zeros((y,x,3))
		self.outBuf = None
		
	def set_r(self,x,y,value):
		self.instance[y][x][0] = value
		
	def set_g(self,x,y,value):
		self.instance[y][x][1] = value
		
	def set_b(self,x,y,value):
		self.instance[y][x][2] = value
		
	def set_rgb(self,x,y,r,g,b):
		self.set_r(x,y,r)
		self.set_g(x,y,g)
		self.set_b(x,y,b)
		
	def set_rgbColor(self,x,y,rgb):
		self.set_r(x,y,rgb[0])
		self.set_r(x,y,rgb[1])
		self.set_r(x,y,rgb[2])
		
	def add_rgbColor(self,x,y,rgb):
		self.instance[y][x][0] += rgb[0]
		self.instance[y][x][1] += rgb[1]
		self.instance[y][x][2] += rgb[2]
		
	def r(self,x,y):
		return self.instance[y][x][0]
		
	def g(self,x,y):
		return self.instance[y][x][1]
		
	def b(self,x,y):
		return self.instance[y][x][2]
		
	def rgb(self,x,y):
		return self.r(x,y),self.g(x,y),self.b(x,y)
				
	def out(self,image_name):
		self.outBuf = Image.fromarray((numpy.clip(self.instance, 0.0, 1.0) * 255).round().astype(numpy.uint8))  
		self.outBuf.save(image_name)
		
def Vector(x,y,z):
	return numpy.array([x, y, z], dtype=numpy.float)
	
def Color(r,g,b):
	return Vector(r,g,b)
	
def normalize(v):
	d = numpy.linalg.norm(v)
	if d==0:
		return v
	return v/d
	
class Vertex(object):
	def __init__(self,pos,normal):
		self.pos = pos
		self.normal = normal

class Camera(object):
	def __init__(self,pos,dir,up):
		self.pos = pos
		self.dir = dir
		self.up = up
	
class Screen(object):
	def __init__(self,camera,setting,scale=30.0,dist=40.0):
		self.dist = dist
		self.width = scale * setting.width / setting.height
		self.height = scale
		self.x = normalize(numpy.cross(camera.dir,camera.up)) * self.width
		self.y = normalize(numpy.cross(camera.dir,self.x)) * self.height
		self.center = camera.pos + camera.dir * self.dist

class Setting(object):
	def __init__(self,w,h,samples,supersamples,max_depth,imagePrefix):
		self.width = w
		self.height = h
		self.samples = samples
		self.supersamples = supersamples
		self.max_depth = max_depth
		self.imagePrefix = imagePrefix
		
class Ray(object):
	def __init__(self,pos,dir):
		self.pos = pos
		self.dir = dir
		
	@staticmethod
	def calculate_dir(camera,screen,setting,x,y,supersample_x,supersample_y):
		rate = 1.0 / setting.supersamples
		rx = supersample_x * rate + rate / 2.0
		ry = supersample_y * rate + rate / 2.0
		end = screen.center + screen.x * ((rx + x) / setting.width - 0.5) + screen.y * ((ry + y) / setting.height - 0.5)
		return normalize(end - camera.pos)
		
class BaseMesh(object):
	def __init__(self,pos):
		self.pos = pos

	def intersect(self,ray):
		print 'not yet implementation'
		# return intersection_vertex
		return None
		
class Sphere(BaseMesh):
	def __init__(self,pos,radius):
		super(Sphere,self).__init__(pos)
		self.radius = radius
		
	def intersect(self,ray):
		intersection = None
		v = ray.pos - self.pos
		B = 2.0 * numpy.dot(ray.dir,v)
		C = numpy.dot(v,v) - pow(self.radius,2)
		discr = pow(B,2) - 4.0 * C
		if discr >= 0.0:
			sqroot = math.sqrt(discr)
			t = (-B - sqroot) * 0.5
			if t < 0.0:
				t = (-B + sqroot) * 0.5
			if t >= 0.0:
				pos = ray.pos + ray.dir * t
				normal = normalize(pos - self.pos)
				intersection = Vertex(pos,normal)
		return intersection
		
class Reflection(object):
	Diffuse = 0
	Specular = 1
	Refraction = 2
	Max = 3
		
class Material(object):
	def __init__(self,color,emission=Color(0,0,0),reflection=Reflection.Diffuse):
		self.color = color
		self.emission = emission
		self.reflection = reflection
		
	@staticmethod
	def diffuse(scene,setting,ray,depth,intersection,orienting_normal):
		w = orienting_normal
		if math.fabs(w[0]) > 0.0000009:
			u = normalize(numpy.cross(Vector(0,1,0),w))
		else:
			u = normalize(numpy.cross(Vector(1,0,0),w))
		v = numpy.cross(w,u)
		r1 = 2.0 * math.pi * random.random()
		r2 = random.random()
		rr2 = math.sqrt(r2)
		ray.pos = intersection.pos
		ray.dir = normalize(u * math.cos(r1) * rr2 + v * math.sin(r1) * rr2 + w * math.sqrt(1.0 - r2))
		return radiance(scene,setting,ray,depth+1)
		
	@staticmethod
	def specular(scene,setting,ray,depth,intersection,orienting_normal):
		ray.pos = intersection.pos
		ray.dir -= intersection.normal * 2.0 * numpy.dot(intersection.normal,ray.dir)
		return radiance(scene,setting,ray,depth+1)
		
	@staticmethod
	def refraction(scene,setting,ray,depth,intersection,orienting_normal):
		into = numpy.dot(orienting_normal,intersection.normal) > 0.0
		default_refraction = 1.0
		object_refraction = 1.5
		if into:
			ray_refraction = default_refraction / object_refraction
		else:
			ray_refraction = object_refraction / default_refraction
		incident_dot = numpy.dot(ray.dir,orienting_normal)
		critical_factor = 1.0 - pow(ray_refraction,2) * (1.0 - pow(incident_dot,2))
				
		reflection_ray = Ray(intersection.pos, ray.dir - intersection.normal * 2.0 * numpy.dot(intersection.normal,ray.dir))
		
		# total reflection
		if critical_factor < 0.0:
			return radiance(scene,setting,reflection_ray,depth+1)
		
		refraction_ray = Ray(intersection.pos, normalize(ray.dir + ray_refraction - intersection.normal * (1.0 if into else -1.0) * incident_dot * ray_refraction + math.sqrt(critical_factor)))
			
		a = object_refraction - default_refraction
		b = object_refraction + default_refraction
		vertical_incidence_factor = pow(a,2) / pow(b,2)
		c = 1.0 - ( -incident_dot if into else numpy.dot(refraction_ray.dir,-orienting_normal) )
		fresnel_incidence_factor = vertical_incidence_factor + (1.0 - vertical_incidence_factor) * pow(c,5)
		radiance_scale = pow(ray_refraction, 2.0)
		refraction_factor = (1.0 - fresnel_incidence_factor) * radiance_scale
		
		probability = 0.75 + fresnel_incidence_factor
		
		if depth > 2:
			if random.random() < probability:
				return radiance(scene,setting,reflection_ray,depth+1) * fresnel_incidence_factor
			else:
				return radiance(scene,setting,refraction_ray,depth+1) * refraction_factor
		else:
			return radiance(scene,setting,reflection_ray,depth+1) * fresnel_incidence_factor + radiance(scene,setting,refraction_ray,depth+1) * refraction_factor
		
Material.radiances = [None] * Reflection.Max
Material.radiances[Reflection.Diffuse] = Material.diffuse
Material.radiances[Reflection.Specular] = Material.specular
Material.radiances[Reflection.Refraction] = Material.refraction		
		
class Geometry(object):
	def __init__(self,mesh,material):
		self.mesh = mesh
		self.material = material
		
class Scene(object):
	def __init__(self,bgColor=Color(0,0,0)):
		self.geometries = []
		self.bgColor = bgColor
		
	def intersect(self,ray):
		dist = float('inf')
		current_dist = float('inf')
		intersection = None
		hit_point = None
		obj_id = None
		for i,g in enumerate(self.geometries):
			hit_point = g.mesh.intersect(ray)
			if hit_point is not None:
				dist = numpy.linalg.norm(hit_point.pos - ray.pos)
				if dist < current_dist:
					intersection = hit_point
					current_dist = dist
					obj_id = i
		return intersection, obj_id
		
def radiance(scene,setting,ray,depth):
	
	intersection, obj_id = scene.intersect(ray)
	if intersection is None:
		return scene.bgColor
		
	geom = scene.geometries[obj_id]
	
	if numpy.dot(intersection.normal,ray.dir) <= 0.0:
		orienting_normal = intersection.normal
	else:
		orienting_normal = -intersection.normal
		
	if depth > setting.max_depth:
		return geom.material.emission
		
	reflect_ratio = float(setting.max_depth - depth) / float(setting.max_depth)
	
	inc_rad = Material.radiances[geom.material.reflection](scene,setting,ray,depth,intersection,orienting_normal)
	weight = geom.material.color * reflect_ratio
	
	return geom.material.emission + weight * inc_rad
	
	
def render(setting,camera,screen,scene,image):
	import datetime
	time_count = 1
	start_time = datetime.datetime.now()
	now_time = None
	
	for y in xrange(setting.height):
		random.seed()
		print 'Rendering (y = " %d ") %.02f%%' % (y,100.0 * y / (setting.height - 1))
		
		for x in xrange(setting.width):
			image.set_rgbColor(x,y,Color(0,0,0))
			
			for sy in xrange(setting.supersamples):
				
				for sx in xrange(setting.supersamples):
					acm_rad = Color(0,0,0)
					
					for s in xrange(setting.samples):
						dir = Ray.calculate_dir(camera,screen,setting,x,y,sx,sy)
						ray = Ray(camera.pos,dir)
						acm_rad += radiance(scene,setting,ray,0) / setting.samples / pow(setting.supersamples,2)
						image.add_rgbColor(x,y,acm_rad)
						
						now_time = datetime.datetime.now() - start_time
						#if now_time.total_seconds > 60.0 * time_count:
						if now_time.seconds > 60.0 * time_count:
							print '%d minute(s)' % time_count
							print 'image_output...'
							image.out(setting.imagePrefix.replace('###','%03d' % time_count))
							time_count += 1

def main():
	import os.path
	import optparse
		
	parser = optparse.OptionParser()
	parser.add_option("-x",dest="w",help="resolution w",default=640)
	parser.add_option("-y",dest="h",help="resolution h",default=480)
	parser.add_option("-s",dest="sample",help="sample",default=4)
	parser.add_option("-p",dest="subpixel",help="subpixel",default=2)
	parser.add_option("-d",dest="max_depth",help="max depth",default=5)
	parser.add_option("-i",dest="image_prefix",help="prefix of image name",default="out")
	options, args = parser.parse_args()
	
	image_name = '%s.###.png' % options.image_prefix
	
	print "width:", options.w
	print "height:", options.h
	print "sample:", options.sample
	print "subpixel:", options.subpixel
	print "max depth:", options.max_depth
	print "image name:", image_name
	
	cd = os.path.dirname(__file__)
	image_name = cd + '/' + image_name
	
	setting = Setting(options.w,options.h,options.sample,options.subpixel,options.max_depth,image_name)
	
	camera = Camera(Vector(50.0, 52.0, 220.0), normalize(Vector(0.0, -0.04, -30.0)), Vector(0.0, 1.0, 0.0))
	screen = Screen(camera,setting)
	
	scene = Scene()
	scene.geometries.append(Geometry(Sphere(Vector( 1e5 +  1, 40.8,        81.6),       1e5), Material(color=Color(0.75, 0.25, 0.25),emission=Color(0, 0, 0),reflection=Reflection.Diffuse)))
	scene.geometries.append(Geometry(Sphere(Vector(-1e5 + 99, 40.8,        81.6),       1e5), Material(color=Color(0.25, 0.25, 0.75),emission=Color(0, 0, 0),reflection=Reflection.Diffuse)))
	scene.geometries.append(Geometry(Sphere(Vector(50,        40.8,        1e5),        1e5), Material(color=Color(0.75, 0.75, 0.75),emission=Color(0, 0, 0),reflection=Reflection.Diffuse)))
	scene.geometries.append(Geometry(Sphere(Vector(50,        40.8,        -1e5 + 250), 1e5), Material(color=Color(0.0,  0.0,  0.0), emission=Color(0, 0, 0),reflection=Reflection.Diffuse)))
	scene.geometries.append(Geometry(Sphere(Vector(50,        1e5,         81.6),       1e5), Material(color=Color(0.75, 0.75, 0.75),emission=Color(0, 0, 0),reflection=Reflection.Diffuse)))
	scene.geometries.append(Geometry(Sphere(Vector(50,        -1e5 + 81.6, 81.6),       1e5), Material(color=Color(0.75, 0.75, 0.75),emission=Color(0, 0, 0),reflection=Reflection.Diffuse)))
	scene.geometries.append(Geometry(Sphere(Vector(65,        20,          20),         20),  Material(color=Color(0.25, 0.75, 0.25),emission=Color(0, 0, 0),reflection=Reflection.Diffuse)))
	scene.geometries.append(Geometry(Sphere(Vector(27,        16.5,        47),         16.5),Material(color=Color(0.99, 0.99, 0.99),emission=Color(0, 0, 0),reflection=Reflection.Specular)))
	scene.geometries.append(Geometry(Sphere(Vector(77,        16.5,        78),         16.5),Material(color=Color(0.99, 0.99, 0.99),emission=Color(0, 0, 0),reflection=Reflection.Refraction)))
	scene.geometries.append(Geometry(Sphere(Vector(50,        90,          81.6),       15.0),Material(color=Color(0.0,  0.0,  0.0), emission=Color(36, 36, 36),reflection=Reflection.Diffuse)))
	
	im = ImageBuffer(options.w,options.h)
	
	render(setting,camera,screen,scene,im)	
	im.out(image_name.replace('###','complete'))

if __name__=='__main__':
	main()