<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name = $_POST["name"];
    $email = $_POST["email"];
    $message = $_POST["message"];

    // Configura la dirección de correo a la que enviar el mensaje
    $to = "victormbar@gmail.com";

    // Configura el asunto del correo
    $subject = "Nuevo mensaje de contacto de $name";

    // Construye el cuerpo del mensaje
    $body = "Nombre: $name\n";
    $body .= "Email: $email\n";
    $body .= "Mensaje:\n$message";

    // Envia el correo
    $headers = "From: $email"; // Puedes personalizar el remitente si es necesario
    mail($to, $subject, $body, $headers);

    // Redirecciona o realiza otras acciones después de enviar el formulario
    header("Location: ../Proyecto/gracias.html");
} else {
    // Si alguien intenta acceder directamente al archivo PHP, redirige a una página de error
    header("Location: ../Proyecto/error.html");
}
?>
