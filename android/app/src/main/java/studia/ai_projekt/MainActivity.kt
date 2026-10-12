package studia.ai_projekt

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import studia.ai_projekt.ui.RpsScreen
import studia.ai_projekt.ui.theme.Ai_projektTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            Ai_projektTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    RpsScreen()
                }
            }
        }
    }
}
