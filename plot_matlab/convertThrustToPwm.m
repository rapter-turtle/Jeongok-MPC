% convertThrustToPwm: u1 -> pwm1
function pwm1 = convertThrustToPwm(thrust)
    if thrust <= 1500
        pwm1 = (thrust - 1450)*0.26; % Any value <= 0 thrust maps to PWM 1000
    else
        pwm1 = (thrust - 1550)*0.26; % Linear conversion
    end
    
    if pwm1 <= 1.3
        pwm1 = 1.3;
    end

end