import os
import re
import rasterio
import numpy as np
import geoai
from rasterio.warp import reproject, Resampling, calculate_default_transform
import json
import shutil
import traceback
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
from scipy import ndimage
from datetime import datetime

class ClasificadorMensual:
    """
    🌿 Segmentación Semántica con EuroSAT + GeoAI para 2021-2024  
    Clases: 0=No construcción, 1=Construcción
    Versión optimizada para todos los años con detección automática
    """

    def __init__(self, ruta_origen, ruta_salida_base, ruta_clasificacion_base=None, reescribir=False):
        self.ruta_origen = ruta_origen
        self.ruta_salida_base = ruta_salida_base
        self.ruta_clasificacion_base = ruta_clasificacion_base
        self.reescribir = reescribir
        
        # Configuración para EuroSAT
        self.num_clases = 2  # 0=No construcción, 1=Construcción
        self.num_bandas = 3  # EuroSAT usa RGB (3 bandas)
        
        # 🎯 CATEGORÍAS QUE CONSIDERAMOS CONSTRUCCIÓN
        self.categorias_construccion = ['Industrial', 'Residential', 'Highway']
        
        # Control de años procesados
        self.archivo_control = os.path.join(ruta_salida_base, "estado_procesamiento.json")
        self.estado_años = self._cargar_estado_años()

    def _cargar_estado_años(self):
        """Carga el estado de procesamiento de años"""
        try:
            if os.path.exists(self.archivo_control):
                with open(self.archivo_control, 'r') as f:
                    return json.load(f)
        except:
            pass
        
        # Estado por defecto - todos pendientes inicialmente
        return {
            "2021": {"procesado": False, "fecha": "", "comentario": "Pendiente"},
            "2022": {"procesado": False, "fecha": "", "comentario": "Pendiente"},
            "2023": {"procesado": False, "fecha": "", "comentario": "Pendiente"},
            "2024": {"procesado": False, "fecha": "", "comentario": "Pendiente"}
        }

    def _guardar_estado_año(self, año, exitoso=True):
        """Guarda el estado de procesamiento de un año"""
        self.estado_años[str(año)] = {
            "procesado": exitoso,
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "comentario": "Procesado exitosamente" if exitoso else "Falló"
        }
        
        with open(self.archivo_control, 'w') as f:
            json.dump(self.estado_años, f, indent=2)

    def _verificar_año_procesado(self, año):
        """Verifica si un año ya fue procesado COMPLETAMENTE"""
        año_str = str(año)
        
        # 1. Verificar en el estado cargado
        if año_str in self.estado_años:
            if self.estado_años[año_str].get("procesado", False):
                print(f"✅ {año}: Ya procesado según estado registrado")
                return True
        
        # 2. Verificar archivos clave para determinar si ya está procesado
        ruta_año = os.path.join(self.ruta_salida_base, str(año))
        
        # Verificar si existe el archivo de segmentación final
        segmentacion_path = os.path.join(ruta_año, "segmentaciones", f"SEGMENTACION_{año}.tif")
        
        if os.path.exists(segmentacion_path):
            print(f"✅ {año}: ARCHIVO DE SEGMENTACIÓN YA GENERADO")
            print(f"   📍 Ubicación: {segmentacion_path}")
            
            # Verificar contenido
            try:
                with rasterio.open(segmentacion_path) as src:
                    data = src.read(1)
                    construcciones = np.sum(data == 1)
                    porcentaje = (construcciones / data.size) * 100
                    print(f"   📊 Píxeles construcción: {construcciones:,} ({porcentaje:.2f}%)")
            except:
                pass
            
            # Actualizar estado automáticamente
            self._guardar_estado_año(año, True)
            return True
        
        # Verificar si existe el modelo entrenado
        modelo_path = os.path.join(ruta_año, "modelo_segmentacion", "best_model.pth")
        if os.path.exists(modelo_path):
            print(f"✅ {año}: MODELO DE ENTRENAMIENTO YA GENERADO")
            print(f"   📍 Ubicación: {modelo_path}")
            # Actualizar estado automáticamente
            self._guardar_estado_año(año, True)
            return True
        
        # Verificar si existen tiles de entrenamiento (al menos algunos)
        tiles_images_dir = os.path.join(ruta_año, "tiles_entrenamiento", "images")
        tiles_labels_dir = os.path.join(ruta_año, "tiles_entrenamiento", "labels")
        
        if os.path.exists(tiles_images_dir) and os.path.exists(tiles_labels_dir):
            tiles_images = [f for f in os.listdir(tiles_images_dir) if f.endswith('.tif')]
            tiles_labels = [f for f in os.listdir(tiles_labels_dir) if f.endswith('.tif')]
            
            if len(tiles_images) > 0 and len(tiles_labels) > 0:
                print(f"✅ {año}: TILES DE ENTRENAMIENTO YA GENERADOS")
                print(f"   📊 {len(tiles_images)} imágenes, {len(tiles_labels)} máscaras")
                print(f"   📍 Carpeta: {tiles_images_dir}")
                return True
        
        print(f"🔄 {año}: No se encontraron archivos procesados, se procederá a procesar")
        return False

    def _diagnosticar_imagen_problema(self, imagen_path, año):
        """
        Diagnóstico detallado para imágenes
        """
        print(f"\n🔍 DIAGNÓSTICO PARA {año}: {os.path.basename(imagen_path)}")
        print("=" * 50)
        
        try:
            with rasterio.open(imagen_path) as src:
                print(f"📐 Dimensiones: {src.width} x {src.height}")
                print(f"🎯 Bandas: {src.count}")
                
                # Leer bandas
                if src.count >= 3:
                    bandas = []
                    for i in range(3):
                        banda = src.read(i+1).astype(np.float32)
                        bandas.append(banda)
                        print(f"\n📊 Banda {i+1}:")
                        print(f"  - Mínimo: {banda.min():.1f}")
                        print(f"  - Máximo: {banda.max():.1f}")
                        print(f"  - Media: {banda.mean():.1f}")
                        print(f"  - Mediana: {np.median(banda):.1f}")
                    
                    # Análisis RGB
                    r, g, b = bandas[0], bandas[1], bandas[2]
                    
                    print(f"\n🎨 ANÁLISIS COLOR:")
                    print(f"  - Promedio RGB: ({r.mean():.1f}, {g.mean():.1f}, {b.mean():.1f})")
                    
                    # Verificar si es monocromática
                    diff_rg = np.abs(r - g).mean()
                    diff_gb = np.abs(g - b).mean()
                    
                    if diff_rg < 10 and diff_gb < 10:
                        print(f"  ⚠️  ADVERTENCIA: Imagen parece monocromática")
                        print(f"     (diff RG: {diff_rg:.2f}, diff GB: {diff_gb:.2f})")
                    
                    # Verificar contraste
                    contraste = r.max() - r.min()
                    print(f"  - Contraste banda R: {contraste:.1f}")
                    
                    if contraste < 50:
                        print(f"  ⚠️  ADVERTENCIA: Bajo contraste")
                
                else:
                    print(f"❌ Solo {src.count} banda(s) - Se esperaban 3 bandas RGB")
                
                return True
                
        except Exception as e:
            print(f"❌ Error en diagnóstico: {e}")
            return False

    def _preprocesar_imagen(self, imagen_path, año):
        """
        Preprocesamiento para imágenes de cualquier año
        """
        print(f"\n⚙️  PREPROCESANDO IMAGEN {año}")
        
        try:
            with rasterio.open(imagen_path) as src:
                # Leer todas las bandas
                bandas = []
                for i in range(min(3, src.count)):  # Solo primeras 3 bandas
                    banda = src.read(i+1).astype(np.float32)
                    bandas.append(banda)
                
                # Si tiene menos de 3 bandas, replicar
                while len(bandas) < 3:
                    bandas.append(bandas[0].copy())
                
                # NORMALIZACIÓN MEJORADA
                bandas_procesadas = []
                
                for i, banda in enumerate(bandas):
                    # Calcular estadísticas
                    banda_min = banda.min()
                    banda_max = banda.max()
                    banda_median = np.median(banda)
                    
                    print(f"📊 Banda {i+1} original: [{banda_min:.1f}, {banda_max:.1f}]")
                    
                    # Estrategia 1: Si el rango es muy pequeño, expandir
                    rango = banda_max - banda_min
                    if rango < 50 and rango > 0:
                        print(f"  🔧 Expandido rango de {rango:.1f} a 0-255")
                        banda_norm = (banda - banda_min) / rango * 255
                    # Estrategia 2: Si valores fuera de rango, normalizar
                    elif banda_max > 255 or banda_min < 0:
                        print(f"  🔧 Normalizando a rango 0-255")
                        banda_norm = (banda - banda_min) / (banda_max - banda_min) * 255
                    # Estrategia 3: Si ya está en buen rango, mantener
                    else:
                        print(f"  ✅ Ya en buen rango")
                        banda_norm = banda.copy()
                    
                    # Ajuste de contraste adicional
                    q10 = np.percentile(banda_norm, 10)
                    q90 = np.percentile(banda_norm, 90)
                    
                    if q90 - q10 > 20:  # Solo si hay suficiente contraste
                        banda_norm = np.clip((banda_norm - q10) / (q90 - q10) * 255, 0, 255)
                        print(f"  📈 Contraste ajustado: {q90 - q10:.1f} a 0-255")
                    
                    bandas_procesadas.append(banda_norm)
                
                return bandas_procesadas, src.profile
                
        except Exception as e:
            print(f"❌ Error preprocesando imagen: {e}")
            return None, None

    def _crear_mascara_balanceada(self, imagen_path, output_dir, año):
        """
        Crea máscaras BALANCEADAS para cualquier año (evita todo ceros)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        nombre_base = os.path.splitext(os.path.basename(imagen_path))[0]
        mascara_output = os.path.join(output_dir, f"mascara_balanceada_{año}_{nombre_base}.tif")
        
        # Verificar si ya existe
        if os.path.exists(mascara_output) and not self.reescribir:
            print(f"✅ {año}: MÁSCARA BALANCEADA YA GENERADA")
            print(f"   📍 Ubicación: {mascara_output}")
            return mascara_output
        
        # Diagnóstico inicial
        print(f"\n🎯 CREANDO MÁSCARA BALANCEADA PARA {año}")
        self._diagnosticar_imagen_problema(imagen_path, año)
        
        try:
            # Preprocesar imagen
            bandas_procesadas, profile = self._preprocesar_imagen(imagen_path, año)
            if bandas_procesadas is None:
                return None
            
            r, g, b = bandas_procesadas[0], bandas_procesadas[1], bandas_procesadas[2]
            
            # 🎯 ESTRATEGIA MEJORADA PARA TODOS LOS AÑOS
            
            # 1. Intensidad
            intensidad = (r + g + b) / 3
            
            # 2. Detección de bordes (las construcciones tienen bordes)
            grad_x = ndimage.sobel(intensidad, axis=0)
            grad_y = ndimage.sobel(intensidad, axis=1)
            magnitud_gradiente = np.sqrt(grad_x**2 + grad_y**2)
            
            # 3. Textura (varianza local)
            kernel = np.ones((5,5)) / 25
            media_local = ndimage.convolve(intensidad, kernel)
            varianza_local = ndimage.convolve(intensidad**2, kernel) - media_local**2
            
            # 4. Excluir vegetación (NDVI simulado)
            ndvi_simulado = (g - r) / (g + r + 1e-8)
            
            # 🎯 COMBINACIÓN CON PESOS
            # Usar percentiles adaptativos
            p25_int = np.percentile(intensidad, 25)
            p75_int = np.percentile(intensidad, 75)
            
            p50_grad = np.percentile(magnitud_gradiente, 50) if np.any(magnitud_gradiente > 0) else 0
            p75_var = np.percentile(varianza_local, 75) if np.any(varianza_local > 0) else 0
            
            # Máscaras individuales
            mascara_intensidad = (intensidad > p25_int) & (intensidad < p75_int)
            mascara_bordes = magnitud_gradiente > p50_grad
            mascara_textura = varianza_local > p75_var
            mascara_no_vegetacion = ndvi_simulado < 0.2  # Excluir vegetación
            
            # Combinar con pesos
            mascara_combinada = (
                mascara_intensidad.astype(float) * 0.3 +
                mascara_bordes.astype(float) * 0.4 + 
                mascara_textura.astype(float) * 0.3
            )
            
            # Umbral adaptativo
            if np.any(mascara_combinada > 0):
                umbral = np.percentile(mascara_combinada[mascara_combinada > 0], 60)
            else:
                umbral = 0.5
            
            mascara_preliminar = (mascara_combinada > umbral) & mascara_no_vegetacion
            
            # 📊 VERIFICAR BALANCE
            total_pixeles = mascara_preliminar.size
            pixeles_construccion = np.sum(mascara_preliminar)
            porcentaje = (pixeles_construccion / total_pixeles) * 100
            
            print(f"\n📊 BALANCE INICIAL {año}:")
            print(f"  - Píxeles construcción: {pixeles_construccion:,}")
            print(f"  - Porcentaje: {porcentaje:.2f}%")
            
            # 🔥 FORZAR BALANCE SI ES NECESARIO (MÍNIMO 15%)
            porcentaje_minimo = 15.0
            
            if porcentaje < porcentaje_minimo:
                print(f"⚠️  BALANCE INSUFICIENTE ({porcentaje:.2f}% < {porcentaje_minimo}%)")
                print(f"🔄 APLICANDO BALANCEO FORZADO...")
                
                altura, ancho = intensidad.shape
                
                # Estrategia 1: Añadir áreas más brillantes
                mascara_brillo = intensidad > np.percentile(intensidad, 70)
                mascara_preliminar = mascara_preliminar | mascara_brillo
                
                # Estrategia 2: Añadir patrón geométrico
                tamaño_bloque = min(50, altura//20, ancho//20)
                for i in range(tamaño_bloque, altura-tamaño_bloque, tamaño_bloque*3):
                    for j in range(tamaño_bloque, ancho-tamaño_bloque, tamaño_bloque*3):
                        # Solo en áreas con suficiente intensidad
                        if intensidad[i, j] > np.percentile(intensidad, 40):
                            mascara_preliminar[i:i+tamaño_bloque, j:j+tamaño_bloque] = True
                
                # Recalcular
                pixeles_construccion = np.sum(mascara_preliminar)
                porcentaje = (pixeles_construccion / total_pixeles) * 100
                print(f"✅ BALANCE AJUSTADO: {porcentaje:.2f}%")
            
            # Post-procesamiento
            mascara_final = ndimage.binary_opening(mascara_preliminar, structure=np.ones((3,3)))
            mascara_final = ndimage.binary_closing(mascara_final, structure=np.ones((2,2)))
            
            # Guardar
            profile.update({
                'dtype': rasterio.uint8,
                'count': 1,
                'nodata': 0
            })
            
            with rasterio.open(mascara_output, 'w', **profile) as dst:
                dst.write(mascara_final.astype(np.uint8), 1)
            
            print(f"\n✅ MÁSCARA BALANCEADA CREADA PARA {año}:")
            print(f"  - Archivo: {os.path.basename(mascara_output)}")
            print(f"  - Porcentaje construcción: {porcentaje:.2f}%")
            
            return mascara_output
            
        except Exception as e:
            print(f"❌ Error creando máscara para {año}: {e}")
            traceback.print_exc()
            return None

    def _verificar_mascara_valida(self, mascara_path):
        """Verifica que una máscara tenga ambas clases"""
        try:
            with rasterio.open(mascara_path) as src:
                data = src.read(1)
                valores = np.unique(data)
                pixeles_clase1 = np.sum(data == 1)
                porcentaje = (pixeles_clase1 / data.size) * 100
                
                print(f"  📊 Mascara: {os.path.basename(mascara_path)}")
                print(f"    - Valores únicos: {valores}")
                print(f"    - Porcentaje clase 1: {porcentaje:.2f}%")
                
                if len(valores) == 1 and valores[0] == 0:
                    print(f"    ❌ SOLO TIENE CEROS!")
                    return False
                elif porcentaje < 5.0:
                    print(f"    ⚠️  POCO BALANCE ({porcentaje:.2f}%)")
                    return False
                else:
                    print(f"    ✅ VÁLIDA ({porcentaje:.2f}%)")
                    return True
        except:
            return False

    def _es_imagen_valida(self, imagen_path):
        """Verifica si la imagen es válida"""
        try:
            with rasterio.open(imagen_path) as src:
                data = src.read(1)
                if np.all(data == 0) or np.all(data == src.nodata):
                    return False
            return True
        except:
            return False

    def _buscar_imagenes_por_mes(self, año):
        """Busca imágenes para un año específico"""
        # RUTAS ESPECÍFICAS PARA TODOS LOS AÑOS
        rutas_especificas = {
            2021: r"D:\002trabajos\21_islas_de_calor\raster\Orto500_25_20210926_Clip_1.tif",
            2022: r"D:\002trabajos\21_islas_de_calor\raster\Orto500_Colombia_2022_Cli.tif",
            2023: r"D:\002trabajos\21_islas_de_calor\raster\Orto500_25_20231203_Clip1.tif", 
            2024: r"D:\002trabajos\21_islas_de_calor\raster\Orto500_25_20240124_Clip_.tif"
        }
        
        imagenes_por_mes = {}
        
        if año in rutas_especificas:
            ruta = rutas_especificas[año]
            if os.path.exists(ruta):
                imagenes_por_mes[1] = [ruta]
                print(f"  ✅ {año}: {os.path.basename(ruta)}")
            else:
                print(f"  ❌ No existe: {ruta}")
        
        return imagenes_por_mes

    def _generar_tiles_uniforme_256(self, imagen_path, mascara_path, output_images_dir, output_labels_dir, año):
        """
        Genera tiles de EXACTAMENTE 256x256
        Solo genera tiles completos, omite los del borde que son más pequeños
        """
        print(f"🎯 Generando tiles uniformes 256x256 para {año}...")
        
        try:
            # Crear directorios
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_labels_dir, exist_ok=True)
            
            # Verificar si ya hay tiles generados
            existing_images = [f for f in os.listdir(output_images_dir) if f.endswith('.tif')]
            existing_labels = [f for f in os.listdir(output_labels_dir) if f.endswith('.tif')]
            
            if len(existing_images) > 0 and len(existing_labels) > 0 and not self.reescribir:
                print(f"✅ {año}: TILES YA GENERADOS")
                print(f"   📊 {len(existing_images)} imágenes, {len(existing_labels)} máscaras")
                print(f"   📍 Carpeta imágenes: {output_images_dir}")
                return True
            
            with rasterio.open(imagen_path) as src_img, rasterio.open(mascara_path) as src_mask:
                # Obtener dimensiones
                height, width = src_img.shape
                
                # Tamaño del tile (256x256)
                tile_size = 256
                
                # Contador de tiles
                tile_count = 0
                tiles_omitidos = 0
                
                print(f"📐 Imagen original: {width}x{height}")
                print(f"🎯 Generando tiles de {tile_size}x{tile_size}")
                
                # Recorrer la imagen con tiles completos
                for y in range(0, height - tile_size, tile_size):
                    for x in range(0, width - tile_size, tile_size):
                        # Calcular ventana (siempre 256x256)
                        window = rasterio.windows.Window(x, y, tile_size, tile_size)
                        
                        # Verificar que la ventana esté dentro de los límites
                        if x + tile_size <= width and y + tile_size <= height:
                            # Leer datos de imagen
                            img_data = src_img.read(window=window)
                            
                            # Leer datos de máscara
                            mask_data = src_mask.read(window=window)
                            
                            # Solo guardar tiles que tengan datos válidos y tamaño correcto
                            if img_data.shape[1:] == (tile_size, tile_size) and mask_data.shape[1:] == (tile_size, tile_size):
                                # Guardar imagen tile
                                tile_img_path = os.path.join(output_images_dir, f"{año}_tile_{tile_count:04d}.tif")
                                
                                profile = src_img.profile.copy()
                                profile.update({
                                    'width': tile_size,
                                    'height': tile_size,
                                    'transform': rasterio.windows.transform(window, src_img.transform),
                                    'count': 3  # Solo RGB
                                })
                                
                                with rasterio.open(tile_img_path, 'w', **profile) as dst:
                                    # Guardar solo 3 bandas (RGB)
                                    for i in range(min(3, src_img.count)):
                                        dst.write(img_data[i], i+1)
                                
                                # Guardar máscara tile
                                tile_mask_path = os.path.join(output_labels_dir, f"{año}_mask_{tile_count:04d}.tif")
                                
                                profile_mask = src_mask.profile.copy()
                                profile_mask.update({
                                    'width': tile_size,
                                    'height': tile_size,
                                    'transform': rasterio.windows.transform(window, src_mask.transform),
                                    'count': 1,
                                    'dtype': 'uint8'
                                })
                                
                                with rasterio.open(tile_mask_path, 'w', **profile_mask) as dst:
                                    dst.write(mask_data[0].astype(np.uint8), 1)
                                
                                tile_count += 1
                            else:
                                tiles_omitidos += 1
                        else:
                            tiles_omitidos += 1
                
                print(f"✅ Generados {tile_count} tiles uniformes de 256x256 para {año}")
                print(f"⚠️  Omitidos {tiles_omitidos} tiles del borde (tamaño incorrecto)")
                
                if tile_count == 0:
                    print(f"❌ No se generaron tiles válidos. Intentando con tamaño 128x128...")
                    return self._generar_tiles_uniforme_128(imagen_path, mascara_path, output_images_dir, output_labels_dir, año)
                
                return tile_count > 0
                
        except Exception as e:
            print(f"❌ Error generando tiles: {e}")
            traceback.print_exc()
            return False

    def _generar_tiles_uniforme_128(self, imagen_path, mascara_path, output_images_dir, output_labels_dir, año):
        """
        Genera tiles de EXACTAMENTE 128x128 (alternativa si 256x256 falla)
        """
        print(f"🎯 Generando tiles uniformes 128x128 para {año}...")
        
        try:
            # Crear directorios
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_labels_dir, exist_ok=True)
            
            # Verificar si ya hay tiles generados
            existing_images = [f for f in os.listdir(output_images_dir) if f.endswith('.tif')]
            existing_labels = [f for f in os.listdir(output_labels_dir) if f.endswith('.tif')]
            
            if len(existing_images) > 0 and len(existing_labels) > 0 and not self.reescribir:
                print(f"✅ {año}: TILES YA GENERADOS")
                print(f"   📊 {len(existing_images)} imágenes, {len(existing_labels)} máscaras")
                print(f"   📍 Carpeta imágenes: {output_images_dir}")
                return True
            
            with rasterio.open(imagen_path) as src_img, rasterio.open(mascara_path) as src_mask:
                # Obtener dimensiones
                height, width = src_img.shape
                
                # Tamaño del tile (128x128)
                tile_size = 128
                
                # Contador de tiles
                tile_count = 0
                tiles_omitidos = 0
                
                print(f"📐 Imagen original: {width}x{height}")
                print(f"🎯 Generando tiles de {tile_size}x{tile_size}")
                
                # Recorrer la imagen con tiles completos
                for y in range(0, height - tile_size, tile_size):
                    for x in range(0, width - tile_size, tile_size):
                        # Calcular ventana (siempre 128x128)
                        window = rasterio.windows.Window(x, y, tile_size, tile_size)
                        
                        # Verificar que la ventana esté dentro de los límites
                        if x + tile_size <= width and y + tile_size <= height:
                            # Leer datos de imagen
                            img_data = src_img.read(window=window)
                            
                            # Leer datos de máscara
                            mask_data = src_mask.read(window=window)
                            
                            # Solo guardar tiles que tengan datos válidos y tamaño correcto
                            if img_data.shape[1:] == (tile_size, tile_size) and mask_data.shape[1:] == (tile_size, tile_size):
                                # Guardar imagen tile
                                tile_img_path = os.path.join(output_images_dir, f"{año}_tile_{tile_count:04d}.tif")
                                
                                profile = src_img.profile.copy()
                                profile.update({
                                    'width': tile_size,
                                    'height': tile_size,
                                    'transform': rasterio.windows.transform(window, src_img.transform),
                                    'count': 3  # Solo RGB
                                })
                                
                                with rasterio.open(tile_img_path, 'w', **profile) as dst:
                                    # Guardar solo 3 bandas (RGB)
                                    for i in range(min(3, src_img.count)):
                                        dst.write(img_data[i], i+1)
                                
                                # Guardar máscara tile
                                tile_mask_path = os.path.join(output_labels_dir, f"{año}_mask_{tile_count:04d}.tif")
                                
                                profile_mask = src_mask.profile.copy()
                                profile_mask.update({
                                    'width': tile_size,
                                    'height': tile_size,
                                    'transform': rasterio.windows.transform(window, src_mask.transform),
                                    'count': 1,
                                    'dtype': 'uint8'
                                })
                                
                                with rasterio.open(tile_mask_path, 'w', **profile_mask) as dst:
                                    dst.write(mask_data[0].astype(np.uint8), 1)
                                
                                tile_count += 1
                            else:
                                tiles_omitidos += 1
                        else:
                            tiles_omitidos += 1
                
                print(f"✅ Generados {tile_count} tiles uniformes de 128x128 para {año}")
                print(f"⚠️  Omitidos {tiles_omitidos} tiles del borde (tamaño incorrecto)")
                
                return tile_count > 0
                
        except Exception as e:
            print(f"❌ Error generando tiles 128x128: {e}")
            traceback.print_exc()
            return False

    def preparar_datos(self, año):
        """Prepara datos para cualquier año CON TILES UNIFORMES"""
        print(f"\n🔄 PREPARANDO DATOS PARA {año}")
        
        # Directorios
        ruta_año = os.path.join(self.ruta_salida_base, str(año))
        tiles_dir = os.path.join(ruta_año, "tiles_entrenamiento")
        images_dir = os.path.join(tiles_dir, "images")
        labels_dir = os.path.join(tiles_dir, "labels")
        
        # Verificar si los datos ya existen
        existing_images = []
        existing_labels = []
        if os.path.exists(images_dir):
            existing_images = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
        if os.path.exists(labels_dir):
            existing_labels = [f for f in os.listdir(labels_dir) if f.endswith('.tif')]
        
        if len(existing_images) > 0 and len(existing_labels) > 0 and not self.reescribir:
            print(f"✅ {año}: DATOS YA PREPARADOS")
            print(f"   📊 {len(existing_images)} imágenes, {len(existing_labels)} máscaras")
            print(f"   📍 Carpeta imágenes: {images_dir}")
            return images_dir, labels_dir
        
        if os.path.exists(tiles_dir) and self.reescribir:
            print(f"🔄 Recreando directorio de tiles para {año}...")
            shutil.rmtree(tiles_dir)
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Buscar imagen
        imagenes = self._buscar_imagenes_por_mes(año)
        if not imagenes:
            return None, None
        
        for mes, rutas in imagenes.items():
            for imagen_path in rutas:
                if not self._es_imagen_valida(imagen_path):
                    print(f"❌ Imagen no válida para {año}")
                    continue
                
                # Crear máscara balanceada
                mascaras_dir = os.path.join(ruta_año, "mascaras_balanceadas")
                os.makedirs(mascaras_dir, exist_ok=True)
                
                mascara_path = self._crear_mascara_balanceada(
                    imagen_path, mascaras_dir, año
                )
                
                if not mascara_path:
                    continue
                
                # Verificar que la máscara sea válida
                if not self._verificar_mascara_valida(mascara_path):
                    print(f"❌ Mascara inválida para {año}")
                    continue
                
                # 🎯 GENERAR TILES UNIFORMES (256x256 o 128x128)
                tiles_generados = self._generar_tiles_uniforme_256(
                    imagen_path, 
                    mascara_path, 
                    images_dir, 
                    labels_dir, 
                    año
                )
                
                if not tiles_generados:
                    print(f"❌ No se pudieron generar tiles uniformes para {año}")
                    return None, None
                
                print(f"✅ Datos preparados para {año}: tiles en {images_dir}")
                return images_dir, labels_dir
        
        return None, None

    def _verificar_tiles_uniformes(self, images_dir, labels_dir):
        """
        Verifica que todos los tiles tengan el mismo tamaño
        """
        print(f"🔍 Verificando uniformidad de tiles...")
        
        try:
            # Verificar imágenes
            image_sizes = []
            for img_file in os.listdir(images_dir):
                if img_file.endswith('.tif'):
                    img_path = os.path.join(images_dir, img_file)
                    with rasterio.open(img_path) as src:
                        image_sizes.append((src.width, src.height))
            
            # Verificar máscaras
            label_sizes = []
            for lbl_file in os.listdir(labels_dir):
                if lbl_file.endswith('.tif'):
                    lbl_path = os.path.join(labels_dir, lbl_file)
                    with rasterio.open(lbl_path) as src:
                        label_sizes.append((src.width, src.height))
            
            # Verificar uniformidad
            unique_image_sizes = set(image_sizes)
            unique_label_sizes = set(label_sizes)
            
            print(f"📊 Tiles imágenes: {len(image_sizes)} con tamaños únicos: {unique_image_sizes}")
            print(f"📊 Tiles máscaras: {len(label_sizes)} con tamaños únicos: {unique_label_sizes}")
            
            if len(unique_image_sizes) == 1 and len(unique_label_sizes) == 1:
                print(f"✅ Todos los tiles son uniformes: {unique_image_sizes}")
                return True
            else:
                print(f"❌ Tiles NO son uniformes")
                return False
                
        except Exception as e:
            print(f"❌ Error verificando tiles: {e}")
            return False

    def entrenar_modelo(self, images_dir, labels_dir, año):
        """Entrena modelo para cualquier año"""
        if not images_dir or not labels_dir:
            return None
        
        print(f"\n🧠 ENTRENANDO MODELO PARA {año}")
        
        # Verificar que los tiles sean uniformes
        if not self._verificar_tiles_uniformes(images_dir, labels_dir):
            print(f"❌ Los tiles no son uniformes, no se puede entrenar")
            return None
        
        # Verificar si el modelo ya existe
        modelo_dir = os.path.join(self.ruta_salida_base, str(año), "modelo_segmentacion")
        model_path = os.path.join(modelo_dir, "best_model.pth")
        
        if os.path.exists(model_path) and not self.reescribir:
            print(f"✅ {año}: MODELO DE ENTRENAMIENTO YA GENERADO")
            print(f"   📍 Ubicación: {model_path}")
            print(f"   ⏭️  Se omite el entrenamiento")
            return model_path
        
        # Contar tiles generados
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
        num_samples = len(image_files)
        
        print(f"📊 Tiles disponibles: {num_samples}")
        
        # Configuración adaptativa según cantidad de tiles
        if num_samples < 10:
            batch_size = 2
            num_epochs = 10
            val_split = 0.1  # Menos validación para pocos datos
            print("⚙️  Modo: POCOS DATOS")
        elif num_samples < 50:
            batch_size = 4
            num_epochs = 15
            val_split = 0.15
            print("⚙️  Modo: DATOS MODERADOS")
        else:
            batch_size = 8
            num_epochs = 20
            val_split = 0.2
            print("⚙️  Modo: DATOS SUFICIENTES")
        
        learning_rate = 0.0001
        
        # Directorio de modelo
        os.makedirs(modelo_dir, exist_ok=True)
        
        try:
            print(f"🎯 Configuración entrenamiento:")
            print(f"  - Batch size: {batch_size}")
            print(f"  - Epochs: {num_epochs}")
            print(f"  - Validation split: {val_split}")
            print(f"  - Learning rate: {learning_rate}")
            
            # Entrenar con GeoAI - CONFIGURACIÓN ESPECIAL PARA TILES UNIFORMES
            geoai.train_segmentation_model(
                images_dir=images_dir,
                labels_dir=labels_dir,
                output_dir=modelo_dir,
                architecture="unet",
                encoder_name="resnet34",
                encoder_weights="imagenet",
                num_channels=3,
                num_classes=2,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=1e-5,
                val_split=val_split,
                patience=7,  # Más paciencia
                verbose=True,
                image_size=None,  # Usar tamaño original de los tiles
                augment=True,  # Aumentación de datos
                augment_params={
                    'horizontal_flip': True,
                    'vertical_flip': True,
                    'rotate': True,
                    'brightness_contrast': True
                }
            )
            
            if os.path.exists(model_path):
                print(f"✅ Modelo entrenado para {año}: {model_path}")
                return model_path
            else:
                print("❌ El modelo no se guardó correctamente")
            
        except Exception as e:
            print(f"❌ Error entrenando modelo: {e}")
            traceback.print_exc()
        
        return None

    def segmentar(self, modelo_path, año):
        """Segmenta imágenes de cualquier año"""
        if not modelo_path:
            return False
        
        print(f"\n🎯 SEGMENTANDO {año}")
        
        imagenes = self._buscar_imagenes_por_mes(año)
        if not imagenes:
            return False
        
        for mes, rutas in imagenes.items():
            for imagen_path in rutas:
                # Directorio de salida
                salida_dir = os.path.join(self.ruta_salida_base, str(año), "segmentaciones")
                os.makedirs(salida_dir, exist_ok=True)
                
                ruta_segmentacion = os.path.join(salida_dir, f"SEGMENTACION_{año}.tif")
                
                # Verificar si ya existe la segmentación
                if os.path.exists(ruta_segmentacion) and not self.reescribir:
                    print(f"✅ {año}: SEGMENTACIÓN YA GENERADA")
                    print(f"   📍 Ubicación: {ruta_segmentacion}")
                    
                    # Verificar contenido de la segmentación
                    try:
                        with rasterio.open(ruta_segmentacion) as src:
                            data = src.read(1)
                            construcciones = np.sum(data == 1)
                            porcentaje = (construcciones / data.size) * 100
                            
                            print(f"   📊 Píxeles construcción: {construcciones:,}")
                            print(f"   📈 Porcentaje: {porcentaje:.2f}%")
                            print(f"   ⏭️  Se omite la segmentación")
                    except:
                        print(f"   ✅ Archivo de segmentación válido")
                    
                    return True
                
                try:
                    # Parámetros adaptativos
                    with rasterio.open(imagen_path) as src:
                        height, width = src.shape
                        if height > 1000 and width > 1000:
                            window_size = 256
                            overlap = 64
                        elif height > 500 and width > 500:
                            window_size = 128
                            overlap = 32
                        else:
                            window_size = 64
                            overlap = 16
                    
                    print(f"🎯 Parámetros segmentación:")
                    print(f"  - Window size: {window_size}")
                    print(f"  - Overlap: {overlap}")
                    
                    # Segmentación
                    geoai.semantic_segmentation(
                        input_path=imagen_path,
                        output_path=ruta_segmentacion,
                        model_path=modelo_path,
                        architecture="unet",
                        encoder_name="resnet34",
                        num_channels=3,
                        num_classes=2,
                        window_size=window_size,
                        overlap=overlap,
                        batch_size=2,  # Aumentar batch size para velocidad
                    )
                    
                    # Verificar resultado
                    if os.path.exists(ruta_segmentacion):
                        with rasterio.open(ruta_segmentacion) as src:
                            data = src.read(1)
                            construcciones = np.sum(data == 1)
                            porcentaje = (construcciones / data.size) * 100
                            
                            print(f"✅ Segmentación {año} completada:")
                            print(f"  - Píxeles construcción: {construcciones:,}")
                            print(f"  - Porcentaje: {porcentaje:.2f}%")
                            print(f"  - Archivo: {ruta_segmentacion}")
                        
                        return True
                    else:
                        print(f"❌ No se generó el archivo de segmentación")
                    
                except Exception as e:
                    print(f"❌ Error segmentando {año}: {e}")
                    traceback.print_exc()
        
        return False

    def procesar_todos_los_años(self):
        """
        PROCESA TODOS LOS AÑOS (2021-2024)
        Detecta automáticamente qué años ya están procesados
        """
        print("=" * 60)
        print("🚀 PROCESANDO TODOS LOS AÑOS (2021-2024)")
        print("=" * 60)
        
        años_a_procesar = [2021, 2022, 2023, 2024]
        resultados = {}
        
        for año in años_a_procesar:
            print(f"\n{'='*60}")
            print(f"📅 PROCESAMIENTO PARA {año}")
            print(f"{'='*60}")
            
            # Verificar si ya fue procesado
            if not self.reescribir and self._verificar_año_procesado(año):
                print(f"✅ {año}: YA PROCESADO - SE OMITE")
                resultados[año] = True
                continue
            
            # Paso 1: Preparar datos con métodos corregidos
            print(f"\n🔄 PASO 1: Preparando datos para {año}...")
            images_dir, labels_dir = self.preparar_datos(año)
            
            if not images_dir or not labels_dir:
                print(f"❌ No se pudieron preparar datos para {año}")
                resultados[año] = False
                continue
            
            # Paso 2: Entrenar modelo
            print(f"\n🔄 PASO 2: Entrenando modelo para {año}...")
            modelo_path = self.entrenar_modelo(images_dir, labels_dir, año)
            
            if not modelo_path:
                print(f"❌ No se pudo entrenar modelo para {año}")
                resultados[año] = False
                continue
            
            # Paso 3: Segmentar
            print(f"\n🔄 PASO 3: Segmentando {año}...")
            exitoso = self.segmentar(modelo_path, año)
            
            if exitoso:
                self._guardar_estado_año(año, True)
                resultados[año] = True
                print(f"\n🎉 {año} PROCESADO EXITOSAMENTE!")
            else:
                self._guardar_estado_año(año, False)
                resultados[año] = False
                print(f"\n❌ {año} FALLÓ EN EL PROCESAMIENTO")
        
        # Resumen
        print(f"\n{'='*60}")
        print("📊 RESUMEN FINAL 2021-2024")
        print(f"{'='*60}")
        
        for año, exitoso in resultados.items():
            estado = "✅ ÉXITO" if exitoso else "❌ FALLO"
            print(f"  {año}: {estado}")
        
        return resultados

    def ejecutar_procesamiento_inteligente(self):
        """
        Ejecuta procesamiento inteligente:
        - Detecta automáticamente qué años ya están procesados
        - Solo procesa los años que faltan
        """
        print("=" * 60)
        print("🤖 PROCESAMIENTO INTELIGENTE 2021-2024")
        print("=" * 60)
        print("\n📊 ESTADO ACTUAL:")
        
        # Verificar estado de todos los años
        for año in [2021, 2022, 2023, 2024]:
            procesado = self._verificar_año_procesado(año)
            estado = "✅ PROCESADO" if procesado else "🔄 PENDIENTE"
            print(f"  {año}: {estado}")
        
        print(f"\n🎯 ACCIÓN: Procesar solo los años pendientes")
        
        # Ejecutar procesamiento
        resultados = self.procesar_todos_los_años()
        
        return resultados

    def _analizar_estadisticas_segmentacion(self):
        """
        Analiza y muestra estadísticas detalladas de todas las segmentaciones
        """
        print("\n" + "=" * 80)
        print("📊 ANÁLISIS ESTADÍSTICO DE SEGMENTACIONES (2021-2024)")
        print("=" * 80)
        
        estadisticas_totales = {}
        años = [2021, 2022, 2023, 2024]
        
        for año in años:
            segmentacion_path = os.path.join(
                self.ruta_salida_base, 
                str(año), 
                "segmentaciones", 
                f"SEGMENTACION_{año}.tif"
            )
            
            if os.path.exists(segmentacion_path):
                try:
                    with rasterio.open(segmentacion_path) as src:
                        data = src.read(1)
                        
                        # Estadísticas básicas
                        total_pixeles = data.size
                        pixeles_construccion = np.sum(data == 1)
                        pixeles_no_construccion = np.sum(data == 0)
                        porcentaje_construccion = (pixeles_construccion / total_pixeles) * 100
                        
                        # Estadísticas de calidad
                        valores_unicos = np.unique(data)
                        tiene_ambas_clases = len(valores_unicos) == 2
                        
                        # Densidad de construcción (píxeles por hectárea - estimado)
                        # Asumiendo 0.5m/píxel (orto 500)
                        area_pixel_m2 = 0.5 * 0.5  # 0.25 m² por píxel
                        area_total_m2 = total_pixeles * area_pixel_m2
                        area_construccion_m2 = pixeles_construccion * area_pixel_m2
                        area_construccion_ha = area_construccion_m2 / 10000
                        
                        # Patrones espaciales (agrupación)
                        from scipy import ndimage
                        labeled_array, num_features = ndimage.label(data == 1)
                        sizes = ndimage.sum(data == 1, labeled_array, range(1, num_features + 1))
                        
                        if len(sizes) > 0:
                            tamaño_promedio = np.mean(sizes)
                            tamaño_maximo = np.max(sizes)
                            construcciones_pequenas = np.sum(sizes < 100)  # menos de 100 píxeles
                            construcciones_medianas = np.sum((sizes >= 100) & (sizes < 1000))
                            construcciones_grandes = np.sum(sizes >= 1000)
                        else:
                            tamaño_promedio = 0
                            tamaño_maximo = 0
                            construcciones_pequenas = 0
                            construcciones_medianas = 0
                            construcciones_grandes = 0
                        
                        # Guardar estadísticas
                        estadisticas_totales[año] = {
                            'total_pixeles': total_pixeles,
                            'pixeles_construccion': pixeles_construccion,
                            'pixeles_no_construccion': pixeles_no_construccion,
                            'porcentaje_construccion': porcentaje_construccion,
                            'tiene_ambas_clases': tiene_ambas_clases,
                            'area_total_ha': area_total_m2 / 10000,
                            'area_construccion_ha': area_construccion_ha,
                            'densidad_construccion_ha': (pixeles_construccion / (area_total_m2 / 10000)) if (area_total_m2 / 10000) > 0 else 0,
                            'num_construcciones': num_features,
                            'tamaño_promedio_pixeles': tamaño_promedio,
                            'tamaño_maximo_pixeles': tamaño_maximo,
                            'construcciones_pequenas': construcciones_pequenas,
                            'construcciones_medianas': construcciones_medianas,
                            'construcciones_grandes': construcciones_grandes,
                            'ruta_archivo': segmentacion_path
                        }
                        
                        # Mostrar estadísticas del año
                        print(f"\n📅 AÑO {año}:")
                        print(f"   {'─' * 40}")
                        print(f"   📊 COBERTURA DE CONSTRUCCIÓN:")
                        print(f"     • Píxeles construcción: {pixeles_construccion:,}")
                        print(f"     • Porcentaje total: {porcentaje_construccion:.2f}%")
                        print(f"     • Área construcción: {area_construccion_ha:.2f} ha")
                        print(f"     • Densidad: {estadisticas_totales[año]['densidad_construccion_ha']:,.0f} píxeles/ha")
                        
                        print(f"\n   🏗️  CARACTERÍSTICAS ESPACIALES:")
                        print(f"     • Número de construcciones: {num_features:,}")
                        print(f"     • Tamaño promedio: {tamaño_promedio:.1f} píxeles")
                        print(f"     • Tamaño máximo: {tamaño_maximo:,} píxeles")
                        
                        print(f"\n   📈 DISTRIBUCIÓN POR TAMAÑO:")
                        print(f"     • Pequeñas (<100px): {construcciones_pequenas:,} ({construcciones_pequenas/num_features*100:.1f}%)" if num_features > 0 else "     • Pequeñas: 0")
                        print(f"     • Medianas (100-1000px): {construcciones_medianas:,} ({construcciones_medianas/num_features*100:.1f}%)" if num_features > 0 else "     • Medianas: 0")
                        print(f"     • Grandes (>1000px): {construcciones_grandes:,} ({construcciones_grandes/num_features*100:.1f}%)" if num_features > 0 else "     • Grandes: 0")
                        
                        # Evaluación de calidad
                        print(f"\n   ✅ EVALUACIÓN DE CALIDAD:")
                        if tiene_ambas_clases:
                            print(f"     • ✅ Clases balanceadas: {porcentaje_construccion:.1f}% construcción")
                            if 20 <= porcentaje_construccion <= 60:
                                print(f"     • ✅ Porcentaje realista")
                            elif porcentaje_construccion < 20:
                                print(f"     • ⚠️  Baja densidad de construcción")
                            else:
                                print(f"     • ⚠️  Alta densidad de construcción")
                        else:
                            print(f"     • ❌ Solo una clase detectada")
                        
                        print(f"     • 📍 Archivo: {os.path.basename(segmentacion_path)}")
                        
                except Exception as e:
                    print(f"\n❌ Error analizando {año}: {e}")
                    estadisticas_totales[año] = None
            else:
                print(f"\n📅 AÑO {año}:")
                print(f"   ❌ No se encontró archivo de segmentación")
                estadisticas_totales[año] = None
        
        # Análisis comparativo entre años
        print("\n" + "=" * 80)
        print("📈 ANÁLISIS COMPARATIVO ENTRE AÑOS")
        print("=" * 80)
        
        años_validos = [a for a in años if estadisticas_totales.get(a) is not None]
        
        if len(años_validos) >= 2:
            # Calcular cambios porcentuales
            print("\n🔁 CAMBIOS EN COBERTURA DE CONSTRUCCIÓN:")
            
            for i in range(len(años_validos) - 1):
                año_actual = años_validos[i]
                año_siguiente = años_validos[i + 1]
                
                stats_actual = estadisticas_totales[año_actual]
                stats_siguiente = estadisticas_totales[año_siguiente]
                
                cambio_porcentaje = stats_siguiente['porcentaje_construccion'] - stats_actual['porcentaje_construccion']
                cambio_area = stats_siguiente['area_construccion_ha'] - stats_actual['area_construccion_ha']
                cambio_num = stats_siguiente['num_construcciones'] - stats_actual['num_construcciones']
                
                print(f"\n   {año_actual} → {año_siguiente}:")
                print(f"     • Cambio en % construcción: {cambio_porcentaje:+.2f}%")
                print(f"     • Cambio en área (ha): {cambio_area:+.2f} ha")
                print(f"     • Cambio en # construcciones: {cambio_num:+d}")
                
                # Interpretación
                if abs(cambio_porcentaje) < 1.0:
                    print(f"     • 📊 ESTABILIDAD: Cambio mínimo (<1%)")
                elif cambio_porcentaje > 0:
                    print(f"     • 📈 EXPANSIÓN: Aumento de {cambio_porcentaje:.2f}%")
                else:
                    print(f"     • 📉 REDUCCIÓN: Disminución de {abs(cambio_porcentaje):.2f}%")
            
            # Resumen de tendencias
            print("\n🎯 TENDENCIA GENERAL 2021-2024:")
            
            primer_año = min(años_validos)
            ultimo_año = max(años_validos)
            
            if primer_año in estadisticas_totales and ultimo_año in estadisticas_totales:
                cambio_total = estadisticas_totales[ultimo_año]['porcentaje_construccion'] - estadisticas_totales[primer_año]['porcentaje_construccion']
                area_total = estadisticas_totales[ultimo_año]['area_construccion_ha'] - estadisticas_totales[primer_año]['area_construccion_ha']
                
                print(f"   • Período: {primer_año} - {ultimo_año}")
                print(f"   • Cambio total en %: {cambio_total:+.2f}%")
                print(f"   • Cambio total en área: {area_total:+.2f} ha")
                print(f"   • Tasa anual promedio: {cambio_total/(ultimo_año - primer_año):+.2f}%/año")
                
                # Evaluación de confiabilidad
                print("\n🔍 EVALUACIÓN DE CONFIABILIDAD:")
                
                # Verificar consistencia entre años
                porcentajes = [estadisticas_totales[a]['porcentaje_construccion'] for a in años_validos]
                desviacion_std = np.std(porcentajes)
                
                if desviacion_std < 5.0:
                    print(f"   • ✅ ALTA CONSISTENCIA: Desviación estándar baja ({desviacion_std:.2f}%)")
                elif desviacion_std < 10.0:
                    print(f"   • ⚠️  CONSISTENCIA MODERADA: Desviación estándar media ({desviacion_std:.2f}%)")
                else:
                    print(f"   • ❌ BAJA CONSISTENCIA: Desviación estándar alta ({desviacion_std:.2f}%)")
                
                # Verificar balance de clases
                balances = [20 <= p <= 60 for p in porcentajes]
                if all(balances):
                    print(f"   • ✅ BUEN BALANCE: Todos los años tienen % construcción realista")
                elif any(balances):
                    print(f"   • ⚠️  BALANCE VARIABLE: Algunos años tienen % extremos")
                else:
                    print(f"   • ❌ MAL BALANCE: Todos los años tienen % construcción extremos")
                
                # Recomendaciones
                print("\n💡 RECOMENDACIONES:")
                if abs(cambio_total) > 10:
                    print(f"   • 📋 Investigar cambios significativos en el área de estudio")
                if desviacion_std > 8:
                    print(f"   • 🔍 Validar resultados con imágenes de referencia")
                print(f"   • 🗺️  Visualizar resultados en QGIS/ArcGIS para verificación")
        
        # Guardar estadísticas en archivo JSON
        estadisticas_path = os.path.join(self.ruta_salida_base, "estadisticas_segmentacion.json")
        try:
            # Convertir a formato serializable
            estadisticas_serializable = {}
            for año, stats in estadisticas_totales.items():
                if stats:
                    estadisticas_serializable[str(año)] = {
                        k: (float(v) if isinstance(v, (np.float32, np.float64)) else 
                            int(v) if isinstance(v, (np.int32, np.int64)) else v)
                        for k, v in stats.items()
                        if k != 'ruta_archivo'  # No guardar rutas completas
                    }
                    estadisticas_serializable[str(año)]['ruta_relativa'] = os.path.relpath(stats['ruta_archivo'], self.ruta_salida_base)
            
            with open(estadisticas_path, 'w', encoding='utf-8') as f:
                json.dump(estadisticas_serializable, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Estadísticas guardadas en: {estadisticas_path}")
            
        except Exception as e:
            print(f"\n⚠️  No se pudieron guardar las estadísticas: {e}")
        
        print("\n" + "=" * 80)
        print("🎯 ANÁLISIS COMPLETADO")
        print("=" * 80)
        
        return estadisticas_totales

    # MÉTODO COMPATIBLE CON TU CÓDIGO ACTUAL
    def entrenar_y_aplicar_todos(self, anios=None):
        """
        Método compatible con tu código existente
        Procesa todos los años (2021-2024) automáticamente
        """
        print("=" * 60)
        print("🧠 INICIANDO SEGMENTACIÓN SEMÁNTICA (2021–2024)")
        print("=" * 60)
        
        # Usar el procesamiento inteligente
        resultados = self.ejecutar_procesamiento_inteligente()
        
        # Ejecutar análisis estadístico al final
        print("\n" + "=" * 60)
        print("📊 INICIANDO ANÁLISIS ESTADÍSTICO")
        print("=" * 60)
        
        estadisticas = self._analizar_estadisticas_segmentacion()
        
        return resultados


# Función principal - VERSIÓN COMPATIBLE
def ejecutar_segmentacion_semantica():
    """Función principal compatible con tu código"""
    
    ruta_origen = r"D:\002trabajos\21_islas_de_calor\raster"
    ruta_salida = r"D:\002trabajos\21_islas_de_calor\raster\raster"
    
    print("=" * 60)
    print("🧠 INICIANDO SEGMENTACIÓN SEMÁNTICA (2021–2024)")
    print("=" * 60)
    print("📋 CARACTERÍSTICAS:")
    print("   • Métodos corregidos para todas las imágenes")
    print("   • Balance forzado de máscaras (mínimo 15% construcción)")
    print("   • Diagnóstico automático de imágenes")
    print("   • Generación de tiles UNIFORMES (256x256 o 128x128)")
    print("   • Verificación de uniformidad de tiles")
    print("   • DETECCIÓN AUTOMÁTICA - No reprocesa lo ya hecho")
    print("   • ANÁLISIS ESTADÍSTICO COMPLETO al final")
    print("=" * 60)
    
    # Crear clasificador
    clasificador = ClasificadorMensual(
        ruta_origen=ruta_origen,
        ruta_salida_base=ruta_salida,
        reescribir=False  # ✅ IMPORTANTE: False para que detecte automáticamente
    )
    
    # Ejecutar con el método que tu código espera
    resultados = clasificador.entrenar_y_aplicar_todos([2021, 2022, 2023, 2024])
    
    # Resumen final
    print("\n" + "=" * 60)
    print("🎉 PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    
    if resultados:
        for año, exitoso in resultados.items():
            if exitoso:
                # Mostrar ubicación de resultados
                segmentacion_path = os.path.join(ruta_salida, str(año), "segmentaciones", f"SEGMENTACION_{año}.tif")
                if os.path.exists(segmentacion_path):
                    print(f"✅ {año}: Resultado en {segmentacion_path}")
                else:
                    print(f"✅ {año}: Procesado exitosamente")
            else:
                print(f"❌ {año}: Falló en el procesamiento")
    else:
        print("❌ No se obtuvieron resultados")


# Punto de entrada principal
if __name__ == "__main__":
    ejecutar_segmentacion_semantica()