-- Datos deliberadamente heterogéneos: algunos productos sin categoría,
-- sin descripción, o sin stock -- justo el caso que row_to_text debe manejar.

INSERT INTO productos (nombre, categoria, precio, descripcion, stock) VALUES
('Teclado mecánico RGB', 'Periféricos', 89.99,
 'Teclado mecánico con switches azules, retroiluminación RGB personalizable y reposamuñecas incluido.',
 25),
('Mouse inalámbrico', 'Periféricos', 34.50,
 NULL,
 40),
('Monitor 27 pulgadas 4K', NULL, 349.00,
 'Panel IPS, 4K UHD, 144Hz, ideal para diseño gráfico y gaming.',
 0),
('Cable USB-C 2m', 'Accesorios', 9.99,
 NULL,
 150),
('Silla ergonómica', NULL, 259.00,
 NULL,
 0),
('Webcam HD 1080p', 'Periféricos', 45.00,
 'Webcam con enfoque automático y micrófono con cancelación de ruido, ideal para videollamadas.',
 12);